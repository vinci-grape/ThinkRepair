static String getCharsetFromContentType(String contentType) {
    if (contentType == null) return null;
    Matcher m = charsetPattern.matcher(contentType);
    if (m.find()) {
        String charset = m.group(1).trim();
        if (Charset.isSupported(charset)) {
            return charset;
        } else {
            charset = charset.toLowerCase(Locale.ENGLISH);
            if (Charset.isSupported(charset)) {
                return charset;
            } else {
                charset = charset.toUpperCase(Locale.ENGLISH);
                if (Charset.isSupported(charset)) {
                    return charset;
                }
            }
        }
    }
    return null;
}