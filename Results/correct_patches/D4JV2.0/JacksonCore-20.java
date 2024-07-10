public void writeEmbeddedObject(Object object) throws IOException {
    if (object == null) { // Check if input is null
        writeNull();
    } else if (object instanceof byte[]) { // Check if input is a byte array
        writeBinary((byte[]) object);
    } else {
        throw new JsonGenerationException("No native support for writing embedded objects of type " + object.getClass().getName(), this);
    }
}