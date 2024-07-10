private boolean inSpecificScope(String[] targetNames, String[] baseTypes, String[] extraTypes) {
    // https://html.spec.whatwg.org/multipage/parsing.html#has-an-element-in-the-specific-scope

    int bottom = Math.max(0, stack.size() - 1);
    final int top = Math.max(0, bottom - MaxScopeSearchDepth);

    for (int pos = bottom; pos >= top; pos--) {
        final String elName = stack.get(pos).nodeName();
        if (inSorted(elName, targetNames))
            return true;
        if (inSorted(elName, baseTypes))
            return false;
        if (extraTypes != null && inSorted(elName, extraTypes))
            return false;
    }
    //Validate.fail("Should not be reachable"); // would end up false because hitting 'html' at root (basetypes)
    return false;
}