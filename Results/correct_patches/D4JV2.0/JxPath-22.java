public static String getNamespaceURI(Node node) {
    if (node instanceof Document) {
        node = ((Document) node).getDocumentElement();
    }

    Element element = (Element) node;

    String uri = element.getNamespaceURI();
    if (uri == null || uri.isEmpty()) {
        String prefix = getPrefix(node);
        String qname = prefix == null ? "xmlns" : "xmlns:" + prefix;

        Node aNode = node;
        while (aNode != null) {
            if (aNode.getNodeType() == Node.ELEMENT_NODE) {
                Attr attr = ((Element) aNode).getAttributeNode(qname);
                if (attr != null) {
                    String value = attr.getValue();
                    return value.isEmpty() ? null : value;
                }
            }
            aNode = aNode.getParentNode();
        }
        return null;
    }
    return uri;
}