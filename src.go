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

type Graph struct {
	Inputs  [](*Node)
	Outputs [](*Node)
}
