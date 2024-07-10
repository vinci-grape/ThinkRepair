// Fixed Function
public ByteBuffer readToByteBuffer(int max) throws IOException {
    Validate.isTrue(max >= 0, "maxSize must be 0 (unlimited) or larger");
    final boolean localCapped = max > 0; // still possibly capped in total stream
    final int bufferSize = DefaultSize;
    final ByteArrayOutputStream outStream = new ByteArrayOutputStream();

    int totalBytesRead = 0;
    while (true) {
        int bytesRemaining = max - totalBytesRead;
        if (localCapped && bytesRemaining <= 0) {
            break;
        }
        
        int bytesToRead = localCapped ? Math.min(bufferSize, bytesRemaining) : bufferSize;
        
        byte[] readBuffer = new byte[bytesToRead];
        int read = read(readBuffer, 0, bytesToRead);
        
        if (read == -1) {
            break;
        }
        
        outStream.write(readBuffer, 0, read);
        totalBytesRead += read;
        
        if (localCapped && totalBytesRead >= max) {
            break;
        }
    }

    return ByteBuffer.wrap(outStream.toByteArray());
}