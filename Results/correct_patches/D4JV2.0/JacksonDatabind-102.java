public JsonSerializer<?> createContextual(SerializerProvider serializers,
        BeanProperty property) throws JsonMappingException
{
    JsonFormat.Value format = findFormatOverrides(serializers, property, handledType());
    if (format == null) {
        return this;
    }

    // Simple case first: serialize as numeric timestamp?
    if (format.getShape().isNumeric()) {
        return withFormat(Boolean.TRUE, null);
    }

    // First: custom pattern will override things
    if (format.hasPattern()) {
        final Locale loc = format.hasLocale()
                        ? format.getLocale()
                        : serializers.getLocale();
        SimpleDateFormat df = new SimpleDateFormat(format.getPattern(), loc);
        TimeZone tz = format.hasTimeZone() ? format.getTimeZone()
                : serializers.getTimeZone();
        df.setTimeZone(tz);
        return withFormat(Boolean.FALSE, df);
    }

    // Check if any format changes are required
    if (format.hasLocale() || format.hasTimeZone() || format.getShape() == JsonFormat.Shape.STRING) {
        DateFormat df0 = serializers.getConfig().getDateFormat();
        // Jackson's own `StdDateFormat` is quite easy to deal with...
        if (df0 instanceof StdDateFormat) {
            StdDateFormat std = (StdDateFormat) df0;
            if (format.hasLocale()) {
                std = std.withLocale(format.getLocale());
            }
            if (format.hasTimeZone()) {
                std = std.withTimeZone(format.getTimeZone());
            }
            return withFormat(Boolean.FALSE, std);
        }

        // Unfortunately there's no generally usable mechanism for changing `DateFormat` instances (or even clone()ing)
        // So: require it to be `SimpleDateFormat`; can't config other types
        if (!(df0 instanceof SimpleDateFormat)) {
            serializers.reportBadDefinition(handledType(), String.format(
                    "figured `DateFormat` (%s) not a `SimpleDateFormat`; cannot configure `Locale` or `TimeZone`",
                    getClass().getName()));
        }
        SimpleDateFormat df = (SimpleDateFormat) df0;
        if (format.hasLocale()) {
            // Ugh. No way to change `Locale`, create copy; must re-create completely:
            df = new SimpleDateFormat(df.toPattern(), format.getLocale());
        } else {
            df = (SimpleDateFormat) df.clone();
        }
        if (format.hasTimeZone() && !format.getTimeZone().equals(df.getTimeZone())) {
            df.setTimeZone(format.getTimeZone());
        }
        return withFormat(Boolean.FALSE, df);
    }

    return this;
}