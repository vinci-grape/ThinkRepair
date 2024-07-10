private static String cacheString(final char[] charBuf, final String[] stringCache, final int start, final int count) {
    // limit (no cache):
    if (count > maxStringCacheLen)
        return new String(charBuf, start, count);

    int len = Math.min(count, charBuf.length - start);
    if (len <= 0) {
        return "";
    }

    // calculate hash:
    int hash = 0;
    int offset = start;
    for (int i = 0; i < len; i++) {
        hash = 31 * hash + charBuf[offset++];
    }

    // get from cache
    final int index = hash & (stringCache.length - 1);
    String cached = stringCache[index];

    if (cached == null) { // miss, add
        cached = new String(charBuf, start, len);
        stringCache[index] = cached;
    } else { // hashcode hit, check equality
        if (rangeEquals(charBuf, start, len, cached)) { // hit
            return cached;
        } else { // hashcode conflict
            cached = new String(charBuf, start, len);
            stringCache[index] = cached; // update the cache, as recently used strings are more likely to show up again
        }
    }
    return cached;
}