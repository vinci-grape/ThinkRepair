public int parseInto(DateTimeParserBucket bucket, String text, int position) {
    String str = text.substring(position);
    int maxLength = 0;
    String maxId = "";
    for (String id : ALL_IDS) {
        if (str.startsWith(id) && id.length() > maxLength) {
            maxLength = id.length();
            maxId = id;
        }
    }
    
    if (maxId.length() > 0) {
        bucket.setZone(DateTimeZone.forID(maxId));
        return position + maxLength;
    }
    return ~position;
}