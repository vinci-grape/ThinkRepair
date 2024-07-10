public Map<String, Integer> getHeaderMap() {
    if (this.headerMap == null) {
        return null;
    }
    Map<String, Integer> deepCopyMap = new LinkedHashMap<String, Integer>();
    for (Map.Entry<String, Integer> entry : this.headerMap.entrySet()) {
        deepCopyMap.put(entry.getKey(), new Integer(entry.getValue()));
    }
    return deepCopyMap;
}