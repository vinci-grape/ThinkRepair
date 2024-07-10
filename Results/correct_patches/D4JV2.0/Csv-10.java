public CSVPrinter(final Appendable out, final CSVFormat format) throws IOException {
    Assertions.notNull(out, "out");
    Assertions.notNull(format, "format");

    this.out = out;
    this.format = format;
    if (format.getHeader() != null) { // check if header exists
        this.printRecord((Object[]) format.getHeader()); // print header
    }
    this.format.validate();
}