// Fixed Function
@Override
public Element empty() {
    List<Node> nodesToRemove = new ArrayList<>(childNodes);
    for (Node node : nodesToRemove) {
        node.remove(); 
    }
    return this;
}