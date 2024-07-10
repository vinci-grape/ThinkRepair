// Fixed Function
public String text() {
    final StringBuilder accum = StringUtil.borrowBuilder();
    NodeTraversor.traverse(new NodeVisitor() {
        public void head(Node node, int depth) {
            if (node instanceof TextNode) {
                TextNode textNode = (TextNode) node;
                appendNormalisedText(accum, textNode);
            } else if (node instanceof Element) {
                Element element = (Element) node;
                if (accum.length() > 0 &&
                    (element.isBlock() || element.tag.normalName().equals("br")) &&
                    !TextNode.lastCharIsWhitespace(accum))
                    accum.append(' ');
            }
        }

        public void tail(Node node, int depth) {
            if (node instanceof Element) {
                Element element = (Element) node;
                if (element.isBlock() && !TextNode.lastCharIsWhitespace(accum)) // Remove unnecessary check for next sibling being TextNode
                    accum.append(' ');
            }
        }
    }, this);

    return StringUtil.releaseBuilder(accum).trim(); // return the accumulated text trimmed
}