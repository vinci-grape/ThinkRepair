public String generateToolTipFragment(String toolTipText) {
    return " title=\"" + toolTipText.replaceAll("\"", "&quot;") + "\" alt=\"\"";
}