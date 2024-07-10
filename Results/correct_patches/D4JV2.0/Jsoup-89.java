public String setValue(String val) {
    String oldVal;
    if (parent != null) {
        oldVal = parent.get(this.key);
        int i = parent.indexOfKey(this.key);
        if (i != Attributes.NotFound) {
            parent.vals[i] = val;
        }
    } else {
        oldVal = this.val; // Store the old value before setting the new value
    }
    this.val = val;
    return Attributes.checkNotNull(oldVal);
}