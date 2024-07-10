public static boolean isJSIdentifier(String s) {
  int length = s.length();

  if (length == 0 ||
      !(Character.isLetter(s.charAt(0)) || s.charAt(0) == '$' || s.charAt(0) == '_')) { // Fixed Line
    return false;
  }

  for (int i = 1; i < length; i++) {
    if ( 
        !(Character.isLetterOrDigit(s.charAt(i)) || s.charAt(i) == '$' || s.charAt(i) == '_')) { // Fixed Line
      return false;
    }
  }

  return true;
}