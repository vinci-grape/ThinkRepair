// Fixed Function
public ByteBuffer readToByteBuffer(int max) throws IOException {
    Validate.isTrue(max >= 0, "maxSize must be 0 (unlimited) or larger");
    final boolean localCapped = max > 0;
    final int bufferSize = localCapped && max < DefaultSize ? max : DefaultSize;
    final byte[] readBuffer = new byte[bufferSize];
    final ByteArrayOutputStream outStream = new ByteArrayOutputStream(bufferSize);

    int remaining = max > 0 ? max : Integer.MAX_VALUE; // Initialize remaining to max value

    while (true) {
        int toRead = Math.min(bufferSize, remaining); // Read minimum of bufferSize or remaining bytes
        int read = read(readBuffer, 0, toRead);

        if (read == -1) {
            break;
        }

        outStream.write(readBuffer, 0, read);

        if (localCapped) {
            remaining -= read;

            if (remaining <= 0) {
                break;
            }
        }
    }

    return ByteBuffer.wrap(outStream.toByteArray());
}