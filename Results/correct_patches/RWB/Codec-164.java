// Fixed Function
@Override
public final String encode(String name) {
    // Bulletproof for trivial input - NINO
    if (name == null || EMPTY.equalsIgnoreCase(name) || SPACE.equalsIgnoreCase(name) || name.length() == 1) {
        return EMPTY;
    }

    // Preprocessing
    name = cleanName(name);

    // BEGIN: Actual encoding part of the algorithm...
    // 1. Delete all vowels unless the vowel begins the word
    name = name.replaceAll("[^a-zA-Z]", "")
               .replaceAll("(?<!\\b)[aeiouAEIOU]", "");

    // 2. Remove second consonant from any double consonant
    name = name.replaceAll("(.)\\1", "$1");

    // 3. Reduce codex to 6 letters by joining the first 3 and last 3 letters
    if (name.length() > 6) {
        name = name.substring(0, 3) + name.substring(name.length() - 3);
    }

    return name;
}