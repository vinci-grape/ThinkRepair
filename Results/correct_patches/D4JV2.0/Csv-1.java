public int read() throws IOException {
    int current = super.read();
    if (current == '\n') {
        lineCounter++;
        if (lastChar == '\r') {
            // If the last character was a carriage return, don't increment lineCounter again
            lineCounter--;
        }
    } else if (current == '\r') {
        lineCounter++;
        // If the next character is a newline, we'll increment lineCounter again there.
    } 
    lastChar = current;
    return current == -1 ? -1 : lastChar;
}