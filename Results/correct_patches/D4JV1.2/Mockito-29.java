public void describeTo(Description description) {
    description.appendText("same(");
    appendQuoting(description);
    if (wanted != null) {
        description.appendText(wanted.toString());
    }
    else {
        description.appendText("null");
    }
    appendQuoting(description);
    description.appendText(")");
}