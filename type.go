package nngo

import "fmt"

type Op string

const (
	Add      Op = "+"
	Multiply Op = "*"
)

type Node struct {
	Label  string
	Op     Op
	Inputs [](*Node)
	Output *Node
	Val    float64
	Grad   float64
}

func (n *Node) IsOutputSymbol() bool {
	return n.Output == nil && len(n.Inputs) == 1
}

func (n *Node) IsInputSymbol() bool {
	return len(n.Inputs) == 0
}

func (n *Node) ComputeVal() {
	switch n.Op {
	case Add:
		n.Val = Sum(Map(n.Inputs, func(n *Node) float64 {
			return n.Val
		}))
	case Multiply:
		n.Val = n.Inputs[0].Val * n.Inputs[1].Val
	case "":
		if n.IsOutputSymbol() {
			n.Val = n.Inputs[0].Val
		}
	}
}

func newNode(label string, op Op, inputs [](*Node), output *Node) Node {
	return Node{
		Label:  label,
		Op:     op,
		Inputs: inputs,
		Output: output,
	}
}

func AddNode(label string, output *Node, inputs [](*Node)) Node {
	return newNode(label, Add, inputs, output)
}

func MultiplyNode(label string, output *Node, input1 *Node, input2 *Node) Node {
	return newNode(label, Multiply, [](*Node){input1, input2}, output)
}

func InputSymbol(label string, connectedTo *Node) Node {
	return Node{
		Label:  label,
		Output: connectedTo,
	}
}

func OutputSymbol(label string, connectedTo *Node) Node {
	return Node{
		Label:  label,
		Inputs: [](*Node){connectedTo},
	}
}

type Graph struct {
	Inputs        [](*Node)
	Output        *Node
	Intermediates [](*Node)
}

func NewGraph(inputs [](*Node), output *Node, intermediates [](*Node)) Graph {
	return Graph{
		Inputs:        inputs,
		Output:        output,
		Intermediates: intermediates,
	}
}

func (g *Graph) SetInputs(vals []float64) (err error) {

	if len(vals) != len(g.Inputs) {
		err = fmt.Errorf("error list of values must match list of inputs")
		return
	}

	for i, val := range vals {
		g.Inputs[i].Val = val
	}
	return
}

type Module struct {
	Graph
	Params [](*Node)
}

func NewLinear(n int, label string) Module {

	intermediates := make([](Node), n+1)
	inputs := make([](*Node), 2*n+1)

	for i := 0; i < n; i++ {
		node := InputSymbol(fmt.Sprintf("%s-input-%d", label, i), &intermediates[i])
		inputs[i] = &node
		temp := InputSymbol(fmt.Sprintf("%s-param-%d", label, i), &intermediates[i])
		inputs[i+n] = &temp
	}
	bias := InputSymbol(fmt.Sprintf("%s-param-%d", label, n), &intermediates[n])
	inputs[2*n] = &bias

	output := OutputSymbol(fmt.Sprintf("%s-output", label), &intermediates[n])
	ptrs := ToPtrs(intermediates[:n])
	ptrs = append(ptrs, &bias)
	intermediates[n] = AddNode(fmt.Sprintf("%s-add", label), &output, ptrs)

	for i := 0; i < n; i++ {
		node := MultiplyNode(
			fmt.Sprintf("%s-prod-%d", label, i),
			&intermediates[n],
			inputs[i],
			inputs[i+n],
		)
		intermediates[i] = node
	}

	return Module{
		Graph:  NewGraph(inputs, &output, ToPtrs(intermediates)),
		Params: inputs[n:],
	}
}
