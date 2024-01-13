package nngo

type Op string

const (
	Add      Op = "+"
	Multiply Op = "*"
)

type Node struct {
	Op     Op
	Inputs [](*Node)
	Output *Node
}

func NewNode(op Op, inputs [](*Node), output *Node) Node {
	return Node{
		Op:     op,
		Inputs: inputs,
		Output: output,
	}
}

type Graph struct {
	Inputs        [](*Node)
	Outputs       [](*Node)
	Intermediates [](*Node)
}

func InputSymbol(connectedTo *Node) Node {
	return Node{
		Output: connectedTo,
	}
}

func OutputSymbol(connectedTo *Node) Node {
	return Node{
		Inputs: [](*Node){connectedTo},
	}
}
