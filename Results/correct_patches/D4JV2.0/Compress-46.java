private static ZipLong unixTimeToZipLong(long l) {
    final long MAX_SIGNED_32_BIT_INT = ((long)1 << 31) - 1;
    final long MIN_SIGNED_32_BIT_INT = -((long)1 << 31);
    if (l < MIN_SIGNED_32_BIT_INT || l > MAX_SIGNED_32_BIT_INT) {
        throw new IllegalArgumentException("X5455 timestamps must fit in a signed 32 bit integer: " + l);
    }
    return new ZipLong(l);
}