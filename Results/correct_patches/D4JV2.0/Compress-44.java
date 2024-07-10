public ChecksumCalculatingInputStream(final Checksum checksum, final InputStream in) {
    if (checksum == null) {
        throw new NullPointerException("Checksum must not be null");
    }
    if (in == null) {
        throw new NullPointerException("InputStream must not be null");
    }

    this.checksum = checksum;
    this.in = in;
}