private boolean isSafeReplacement(Node node, Node replacement) {
  // No checks are needed for simple names.
  if (node.isName()) {
    return true;
  }
  Preconditions.checkArgument(node.isGetProp());

  Node firstChild = node.getFirstChild();
  if (firstChild != null && firstChild.isName()
      && isNameAssignedTo(firstChild.getString(), replacement)) {
    return false;
  } else if (firstChild == null || !firstChild.isGetProp()) {
    return true;
  }

  return isSafeReplacement(firstChild, replacement);
}