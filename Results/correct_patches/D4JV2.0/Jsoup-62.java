boolean anyOtherEndTag(Token t, HtmlTreeBuilder tb) {
    String name = t.asEndTag().name(); // Use "name()" instead of "normalName()"
    ArrayList<Element> stack = tb.getStack();
    if (name != null) { // Check if name is not null
        for (int pos = stack.size() -1; pos >= 0; pos--) {
            Element node = stack.get(pos);
            if (node.nodeName().equals(name)) {
                tb.generateImpliedEndTags(name);
                if (!name.equals(tb.currentElement().nodeName()))
                    tb.error(this);
                tb.popStackToClose(name);
                return true;
            } else {
                if (tb.isSpecial(node)) {
                    tb.error(this);
                    return false;
                }
            }
        }
    }
    return true;
}