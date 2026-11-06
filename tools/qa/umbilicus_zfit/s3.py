"""Anonymous HTTPS access to the public vesuvius-challenge-open-data bucket (no s3fs needed)."""
import time, urllib.request, urllib.parse, xml.etree.ElementTree as ET
B = 'https://vesuvius-challenge-open-data.s3.amazonaws.com'
NS = '{http://s3.amazonaws.com/doc/2006-03-01/}'

def ls(prefix, delimiter='/'):
    """Return (common prefixes, [(key, size)]) under prefix."""
    pre, keys, token = [], [], None
    while True:
        q = {'list-type': '2', 'prefix': prefix, 'delimiter': delimiter}
        if token: q['continuation-token'] = token
        r = ET.fromstring(get(B + '/?' + urllib.parse.urlencode(q)))
        pre += [p.find(NS + 'Prefix').text for p in r.findall(NS + 'CommonPrefixes')]
        keys += [(c.find(NS + 'Key').text, int(c.find(NS + 'Size').text)) for c in r.findall(NS + 'Contents')]
        t = r.find(NS + 'NextContinuationToken')
        if t is None: return pre, keys
        token = t.text

def get(url, tries=5):
    for a in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as f: return f.read()
        except Exception:
            if a == tries - 1: raise
            time.sleep(3 * (a + 1))

def cat(key): return get(B + '/' + urllib.parse.quote(key))
