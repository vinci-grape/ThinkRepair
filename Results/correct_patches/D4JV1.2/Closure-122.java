private void handleBlockComment(Comment comment) {
  if (comment.getValue().matches("(?s).*\\/\\*\\s*@.*|(?s).*\\n\\s*\\*\\s*@.*")) { // Fixed Line
    errorReporter.warning(
        SUSPICIOUS_COMMENT_WARNING,
        sourceName,
        comment.getLineno(), "", 0);
  }
}