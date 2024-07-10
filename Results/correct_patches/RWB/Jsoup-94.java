// Fixed Function
protected void replaceChild(Node out, Node in) {
    Validate.isTrue(out.parentNode == this);
    Validate.notNull(in);
  
    final int index = out.siblingIndex;
  
    out.parentNode = null; // Update the parent node reference for the replaced node 'out' to null
  
    ensureChildNodes().set(index, in); // Replace the old node with the new node at the same index
    in.parentNode = this; // Update the parent node for the new node 'in'
    in.setSiblingIndex(index); // Update the sibling index for the new node 'in'
}