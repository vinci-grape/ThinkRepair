public int writeValue() {
    // Most likely, object:
    if (_type == TYPE_OBJECT) {
        if (_gotName) {
            _gotName = false;
            ++_index;
            return STATUS_OK_AFTER_COLON;
        } else {
            _gotName = true;
            return STATUS_EXPECT_NAME;
        }
    }

    // Ok, array?
    if (_type == TYPE_ARRAY) {
        int ix = _index;
        ++_index;
        return (ix < 0) ? STATUS_OK_AS_IS : STATUS_OK_AFTER_COMMA;
    }
    
    // Nope, root context
    // No commas within root context, but need space
    ++_index;
    return (_index == 0) ? STATUS_OK_AS_IS : STATUS_OK_AFTER_SPACE;
}