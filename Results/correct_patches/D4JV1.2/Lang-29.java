static int toJavaVersionInt(String version) {
    int[] versionArray = toJavaVersionIntArray(version, JAVA_VERSION_TRIM_SIZE);
    int versionInt = toVersionInt(versionArray);
    return versionInt;
}