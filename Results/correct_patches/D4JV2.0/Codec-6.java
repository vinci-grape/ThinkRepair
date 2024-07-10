public int read(byte b[], int offset, int len) throws IOException {
    if (b == null) {
        throw new NullPointerException();
    } else if (offset < 0 || len < 0) {
        throw new IndexOutOfBoundsException();
    } else if (offset > b.length || offset + len > b.length) {
        throw new IndexOutOfBoundsException();
    } else if (len == 0) {
        return 0;
    } else {
        boolean hasMoreData = true;
        int bytesRead = 0;
        while (hasMoreData && bytesRead == 0) {
            if (!base64.hasData()) {
                byte[] buf = new byte[doEncode ? 4096 : 8192];
                int c = in.read(buf);
                if (c == -1) {
                    hasMoreData = false; // added flag to track end of stream
                }
                // A little optimization to avoid System.arraycopy()
                // when possible.
                if (c > 0 && b.length == len) {
                    base64.setInitialBuffer(b, offset, len);
                }
                if (doEncode) {
                    base64.encode(buf, 0, c);
                } else {
                    base64.decode(buf, 0, c);
                }
            }
            bytesRead = base64.readResults(b, offset, len);
        }
        return bytesRead;
    }
}