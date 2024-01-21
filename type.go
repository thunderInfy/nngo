package nngo

import (
	"fmt"
	"math/rand"
)

type Op string

const (
	Add      Op = "+"
	Multiply Op = "*"
)

type Node struct {
	Label   string
	Op      Op
	Inputs  [](*Node)
	Outputs [](*Node)
	Val     float64
	Grad    float64
}

func (n *Node) IsOutputSymbol() bool {
	return n.Outputs == nil && len(n.Inputs) == 1
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

func newNode(label string, op Op, inputs, outputs [](*Node)) Node {
	return Node{
		Label:   label,
		Op:      op,
		Inputs:  inputs,
		Outputs: outputs,
	}
}

func AddNode(label string, outputs, inputs [](*Node)) Node {
	return newNode(label, Add, inputs, outputs)
}

func MultiplyNode(label string, outputs [](*Node), input1 *Node, input2 *Node) Node {
	return newNode(label, Multiply, [](*Node){input1, input2}, outputs)
}

func InputSymbol(label string, connectedTo [](*Node)) Node {
	return Node{
		Label:   label,
		Outputs: connectedTo,
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

func (g *Graph) ZeroGrad() {
	for i := range g.Inputs {
		g.Inputs[i].Grad = 0
	}
	for i := range g.Intermediates {
		g.Intermediates[i].Grad = 0
	}
	g.Output.Grad = 0
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
	Graph  Graph
	Params [](*Node)
}

func (m *Module) Forward(inputValues []float64, optimizer *Optimizer) (err error) {
	inputValues = append(inputValues, optimizer.GetWeights()...)
	err = m.Graph.Forward(inputValues)
	return
}

func (m *Module) Backprop(upstreamGrad float64, optimizer *Optimizer) {
	m.Graph.Backprop(upstreamGrad)
	grads := make([]float64, len(m.Params))
	for i := range grads {
		grads[i] = m.Params[i].Grad
	}
	optimizer.UpdateWeights(grads)
}

func NewLinear(n int, label string) Module {

	intermediates := make([](Node), n+1)
	inputs := make([](*Node), 2*n+1)

	for i := 0; i < n; i++ {
		node := InputSymbol(fmt.Sprintf("%s-input-%d", label, i), [](*Node){&intermediates[i]})
		inputs[i] = &node
		temp := InputSymbol(fmt.Sprintf("%s-param-%d", label, i), [](*Node){&intermediates[i]})
		inputs[i+n] = &temp
	}
	bias := InputSymbol(fmt.Sprintf("%s-param-%d", label, n), [](*Node){&intermediates[n]})
	inputs[2*n] = &bias

	output := OutputSymbol(fmt.Sprintf("%s-output", label), &intermediates[n])
	ptrs := ToPtrs(intermediates[:n])
	ptrs = append(ptrs, &bias)
	intermediates[n] = AddNode(fmt.Sprintf("%s-add", label), [](*Node){&output}, ptrs)

	for i := 0; i < n; i++ {
		node := MultiplyNode(
			fmt.Sprintf("%s-prod-%d", label, i),
			[](*Node){&intermediates[n]},
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

type Optimizer struct {
	NumParams    int
	LearningRate float64
	RandomSource *rand.Rand
	params       []float64
}

func NewOptimizer(numParams int, learningRate float64, randSource *rand.Rand) Optimizer {
	return Optimizer{
		NumParams:    numParams,
		LearningRate: learningRate,
		RandomSource: randSource,
	}
}

func (o *Optimizer) GetWeights() []float64 {
	if len(o.params) == 0 {
		// initialize
		for i := 0; i < o.NumParams; i++ {
			o.params = append(o.params, RandomFloat64(o.RandomSource, -1, 1))
		}
	}
	return o.params
}

func (o *Optimizer) UpdateWeights(grads []float64) {
	for i := 0; i < o.NumParams; i++ {
		o.params[i] -= o.LearningRate * grads[i]
	}
}

type Set[T comparable] map[T]bool

type Stack[T any] struct {
	data []T
}

// Push adds an element to the top of the stack.
func (s *Stack[T]) Push(item T) {
	s.data = append(s.data, item)
}

// Pop removes and returns the top element from the stack.
func (s *Stack[T]) Pop() (t T, empty bool) {
	if len(s.data) == 0 {
		empty = true
		return
	}
	lastIndex := len(s.data) - 1
	t = s.data[lastIndex]
	s.data = s.data[:lastIndex]
	return
}

// Peek returns the top element of the stack without removing it.
func (s *Stack[T]) Peek() (t T, err error) {
	if len(s.data) == 0 {
		err = fmt.Errorf("Stack is empty")
		return
	}
	return s.data[len(s.data)-1], nil
}

// IsEmpty checks if the stack is empty.
func (s *Stack[T]) IsEmpty() bool {
	return len(s.data) == 0
}

// Size returns the number of elements in the stack.
func (s *Stack[T]) Size() int {
	return len(s.data)
}
