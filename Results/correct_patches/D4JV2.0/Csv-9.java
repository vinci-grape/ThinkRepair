<M extends Map<String, String>> M putIn(final M map) {
    if (mapping == null) { // Null check for mapping
        return map;
    }
    for (final Entry<String, Integer> entry : mapping.entrySet()) {
        final int col = entry.getValue();
        if (col < values.length && col >= 0 && !map.containsKey(entry.getKey())) {
            map.put(entry.getKey(), values[col]);
        }
    }
    return map;
}