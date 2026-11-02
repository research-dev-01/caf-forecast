"""
Numba PCAP kernel and resumable gzip/HTTP stream.
Corresponds to Cell 4 of cmsnts_latest.ipynb.
"""

import os, sys, io, re, json, time, zlib, urllib.request, urllib.error
from pathlib import Path
import numba
import numpy as np

from config import BIN_MS, DATA_CACHE, CAPTURE_SECONDS
from utils.helpers import mawi_url, stamp

ACC_NAMES = ['packets', 'wire_bytes', 'wire_bytes_sq', 'tcp_bytes', 'udp_bytes', 'tcp_syn', 'tcp_fin_rst',
             'small_pkts', 'large_pkts']
NACC = len(ACC_NAMES)
PARSER_VERSION = 'nb2'
# state vector: 0 init,1 first tick,2 origin bin,3 previous tick,4 max tick,5 packets,6 out-of-order,7 max backward,
# 8 captured>original,9 captured>snaplen,10 zero original length,11 outside buffer,12 tcp headers parsed,13 non-IP
NST = 16

@numba.njit(nogil=True)
def _u32(b, i, big):
    if big:
        return (np.int64(b[i]) << 24) | (np.int64(b[i + 1]) << 16) | (np.int64(b[i + 2]) << 8) | np.int64(b[i + 3])
    return np.int64(b[i]) | (np.int64(b[i + 1]) << 8) | (np.int64(b[i + 2]) << 16) | (np.int64(b[i + 3]) << 24)

@numba.njit(nogil=True)
def parse_records(buf, pos, valid, big, denom, bin_ticks, snaplen, linktype, stop_ticks, pre_margin, acc, st):
    nb = acc.shape[0]
    while pos + 16 <= valid:
        sec = _u32(buf, pos, big); sub = _u32(buf, pos + 4, big)
        incl = _u32(buf, pos + 8, big); wire = _u32(buf, pos + 12, big)
        if incl > 262144 or sub >= denom:
            return pos, -1
        if pos + 16 + incl > valid:
            break
        tick = sec * denom + sub
        if st[0] == 0:
            st[0] = 1; st[1] = tick; st[2] = tick // bin_ticks - pre_margin; st[3] = tick; st[4] = tick
        if stop_ticks > 0 and tick >= st[1] + stop_ticks:
            return pos, 1
        if tick < st[3]:
            st[6] += 1
            back = st[3] - tick
            if back > st[7]:
                st[7] = back
            if back > 60 * denom:
                return pos, -2
        elif tick - st[3] > 60 * denom:
            return pos, -3
        st[3] = tick
        if tick > st[4]:
            st[4] = tick
        st[5] += 1
        if incl > wire:
            st[8] += 1
        if incl > snaplen:
            st[9] += 1
        if wire == 0:
            st[10] += 1
        idx = tick // bin_ticks - st[2]
        if idx < 0 or idx >= nb:
            st[11] += 1
        else:
            acc[idx, 0] += 1; acc[idx, 1] += wire; acc[idx, 2] += wire * wire
            if wire <= 128:
                acc[idx, 7] += 1
            elif wire >= 1400:
                acc[idx, 8] += 1
            p = pos + 16; o = -1; et = -1
            if linktype == 1 and incl >= 14:
                et = (np.int64(buf[p + 12]) << 8) | np.int64(buf[p + 13]); o = 14; k = 0
                while (et == 0x8100 or et == 0x88A8) and k < 2 and incl >= o + 4:
                    et = (np.int64(buf[p + o + 2]) << 8) | np.int64(buf[p + o + 3]); o += 4; k += 1
            elif (linktype == 101 or linktype == 228 or linktype == 229) and incl >= 1:
                v = np.int64(buf[p]) >> 4; o = 0
                if v == 4:
                    et = 0x0800
                elif v == 6:
                    et = 0x86DD
            proto = -1; l4 = -1
            if et == 0x0800 and incl >= o + 20:
                ihl = (np.int64(buf[p + o]) & 15) * 4
                frag = ((np.int64(buf[p + o + 6]) & 0x1F) << 8) | np.int64(buf[p + o + 7])
                proto = np.int64(buf[p + o + 9])
                if frag == 0:
                    l4 = o + ihl
            elif et == 0x86DD and incl >= o + 40:
                proto = np.int64(buf[p + o + 6]); l4 = o + 40
            else:
                st[13] += 1
            if proto == 6:
                acc[idx, 3] += wire
                if l4 >= 0 and incl >= l4 + 14:
                    fl = np.int64(buf[p + l4 + 13]); st[12] += 1
                    if (fl & 2) != 0 and (fl & 16) == 0:
                        acc[idx, 5] += 1
                    if (fl & 5) != 0:
                        acc[idx, 6] += 1
            elif proto == 17:
                acc[idx, 4] += wire
        pos += 16 + incl
    return pos, 0

# compile once in the main thread before worker threads use the kernel
parse_records(np.zeros(64, np.uint8), 0, 0, 0, 1_000_000, 10_000, 96, 1, 0, 10, np.zeros((4, NACC), np.int64),
              np.zeros(NST, np.int64))

class GzipHTTPStream:
    """Yields decompressed bytes of a .gz URL; resumes broken transfers with HTTP Range, keeping inflate state."""
    def __init__(self, url, read_size=1 << 20, max_retries=8, timeout=120):
        self.url, self.read_size, self.max_retries, self.timeout = url, read_size, max_retries, timeout
        self.received = 0; self.sha = hashlib.sha256(); self.eof = False; self.retries = 0

    def chunks(self):
        d = zlib.decompressobj(31)
        while True:
            headers = {'User-Agent': 'NCC2027-research/2.0 (academic traffic-forecasting study)'}
            if self.received:
                headers['Range'] = f'bytes={self.received}-'
            try:
                with urllib.request.urlopen(urllib.request.Request(self.url, headers=headers), timeout=self.timeout) as r:
                    if self.received and getattr(r, 'status', None) != 206:
                        raise IOError('server ignored the Range request; cannot resume exactly')
                    length = r.headers.get('Content-Length')
                    expected = self.received + int(length) if length else None
                    while True:
                        data = r.read(self.read_size)
                        if not data:
                            if expected is not None and self.received < expected:
                                raise IOError('connection closed early')
                            self.eof = d.eof
                            return
                        self.received += len(data); self.sha.update(data)
                        out = d.decompress(data)
                        while d.eof and d.unused_data:
                            rest = d.unused_data
                            if not rest.strip(b'\x00'):
                                break
                            d = zlib.decompressobj(31); out += d.decompress(rest)
                        if out:
                            yield out
            except (OSError, http.client.HTTPException) as e:
                if isinstance(e, urllib.error.HTTPError) and e.code in (403, 404):
                    raise
                if isinstance(getattr(e, 'reason', e), ssl.SSLCertVerificationError):
                    raise                                  # not transient: the caller falls back to the mirror
                self.retries += 1
                if self.retries > self.max_retries:
                    raise
                print(f'  stream error ({type(e).__name__}: {str(e)[:80]}); resuming at byte {self.received:,}')
                time.sleep(min(60, 5 * self.retries))

def mawi_published_packets(date):
    try:
        html = urllib.request.urlopen(mawi_url(date).replace('.pcap.gz', '.html'), timeout=30).read().decode('latin1')
        m = re.search(r'# of packets:</b>\s*([0-9]+)', html)
        return int(m.group(1)) if m else None
    except Exception:
        return None

def acquire_trace(date, capture_seconds=CAPTURE_SECONDS, bin_ms=BIN_MS, verbose=True, mirror=False):
    cache = DATA_CACHE / f'{date}_{bin_ms}ms_{capture_seconds}s_{PARSER_VERSION}.npz'
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        return z['acc'], json.loads(str(z['meta']))
    url = mawi_url(date, mirror); stream = GzipHTTPStream(url); gen = stream.chunks()
    buf = np.empty(1 << 25, np.uint8); valid = 0; hdr = None; code = 0
    pre_margin = int(round(2000 / bin_ms))
    span = capture_seconds if capture_seconds > 0 else 1000
    acc = np.zeros((int(span * 1000 / bin_ms) + 2 * pre_margin + 16, NACC), np.int64)
    st = np.zeros(NST, np.int64); t0 = last = time.time()
    try:
        for out in gen:
            a = np.frombuffer(out, np.uint8)
            if valid + a.size > buf.size:
                nbuf = np.empty(max(2 * buf.size, valid + a.size), np.uint8); nbuf[:valid] = buf[:valid]; buf = nbuf
            buf[valid:valid + a.size] = a; valid += a.size
            pos = 0
            if hdr is None:
                if valid < 24:
                    continue
                magic = bytes(buf[:4])
                table = {b'\xd4\xc3\xb2\xa1': (0, 1_000_000), b'\xa1\xb2\xc3\xd4': (1, 1_000_000),
                         b'\x4d\x3c\xb2\xa1': (0, 1_000_000_000), b'\xa1\xb2\x3c\x4d': (1, 1_000_000_000)}
                if magic not in table:
                    raise ValueError(f'{date}: not a classic PCAP (magic {magic.hex()})')
                big, denom = table[magic]
                snaplen, linktype = np.frombuffer(bytes(buf[16:24]), dtype='>u4' if big else '<u4').tolist()
                bin_ticks = denom * bin_ms // 1000; stop_ticks = capture_seconds * denom
                hdr = dict(big=big, denom=denom, snaplen=int(snaplen), linktype=int(linktype)); pos = 24
            pos, code = parse_records(buf, pos, valid, big, denom, bin_ticks, hdr['snaplen'], hdr['linktype'],
                                      stop_ticks, pre_margin, acc, st)
            if code != 0:
                break
            rem = valid - pos; buf[:rem] = buf[pos:valid].copy(); valid = rem
            if verbose and time.time() - last > 60:
                last = time.time()
                stamp(f'  {date}: {stream.received / 2**20:,.0f} MiB read, {st[5]:,} packets, '
                      f'{(st[4] - st[1]) / denom:.0f} s of trace, {stream.received / 2**20 / (last - t0):.1f} MiB/s')
    finally:
        gen.close()
    if code < 0:
        raise ValueError(f'{date}: ' + {-1: 'unparsable record header', -2: 'timestamp >60 s backwards',
                                        -3: 'timestamp >60 s forward gap'}[code])
    if hdr is None:
        raise ValueError(f'{date}: empty stream')
    denom = hdr['denom']; bin_ticks = denom * bin_ms // 1000
    if capture_seconds > 0:
        if code != 1:
            raise ValueError(f'{date}: capture shorter than {capture_seconds} s')
        end = (st[1] + capture_seconds * denom) // bin_ticks - st[2]
    else:
        if not stream.eof or valid != 0:
            raise ValueError(f'{date}: gzip/PCAP stream incomplete')
        end = st[4] // bin_ticks - st[2]
    start = pre_margin + 1
    series = acc[start:end].copy()
    published = mawi_published_packets(date)
    meta = dict(date=date, url=url, capture_seconds_requested=capture_seconds, bin_ms=bin_ms,
                parser=PARSER_VERSION, packets_parsed=int(st[5]), packets_in_series=int(series[:, 0].sum()),
                packets_before_first_full_bin=int(acc[:start, 0].sum()), bins=int(len(series)),
                seconds=len(series) * bin_ms / 1000, timestamp_denominator=denom, snaplen=hdr['snaplen'],
                linktype=hdr['linktype'], out_of_order_packets=int(st[6]), max_backward_jump_s=float(st[7] / denom),
                captured_over_original=int(st[8]), captured_over_snaplen=int(st[9]),
                zero_original_length=int(st[10]), outside_bin_buffer=int(st[11]), tcp_headers_parsed=int(st[12]),
                non_ip_packets=int(st[13]), compressed_bytes_read=int(stream.received),
                sha256_of_bytes_read=stream.sha.hexdigest(), gzip_eof_reached=bool(stream.eof),
                http_resumes=int(stream.retries), mawi_published_packets=published,
                full_capture_packet_count_matches=(published == int(st[5])) if capture_seconds == 0 else None,
                wall_seconds=round(time.time() - t0, 1))
    assert meta['outside_bin_buffer'] == 0, meta
    np.savez_compressed(cache, acc=series, meta=json.dumps(meta))
    return series, meta