private boolean testValidProtocol(Element el, Attribute attr, Set<Protocol> protocols) {
    String value = el.absUrl(attr.getKey());
    if (value.equals("")) {
        value = attr.getValue(); // return the original value if it is not possible to create an absolute url
    } else if (!el.hasAttr("abs:" + attr.getKey())) { // check if the URL is relative
        return false;
    }
    if (!preserveRelativeLinks) {
        attr.setValue(value);
    }
    for (Protocol protocol : protocols) {
        String prot = protocol.toString() + ":";
        if (value.toLowerCase().startsWith(prot)) {
            return true;
        }
    }
    return false;
}