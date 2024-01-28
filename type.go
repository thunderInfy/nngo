package nngo

import (
	"fmt"
	"math"
	"math/rand"
)

type Op string

const (
	Add        Op = "+"
	Multiply   Op = "*"
	Relu       Op = "relu"
	Exp        Op = "exp"
	Dot        Op = "dot"
	Reciprocal Op = "reciprocal"
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

func (n *Node) ComputeGrad() {
	switch n.Op {
	case Add:
		for _, inp := range n.Inputs {
			inp.Grad += n.Grad
		}
	case Multiply:
		for _, inp := range n.Inputs {
			if inp.Val != 0 {
				inp.Grad += (n.Grad * n.Val) / (inp.Val)
			}
		}
	case Relu:
		inp := n.Inputs[0]
		if inp.Val > 0 {
			inp.Grad += n.Grad
		}
	case Exp:
		inp := n.Inputs[0]
		inp.Grad += n.Grad * n.Val
	case Dot:
		d := len(n.Inputs) / 2
		for i := range n.Inputs {
			if i < d {
				n.Inputs[i].Grad += n.Grad * n.Inputs[i+d].Val
			} else {
				n.Inputs[i].Grad += n.Grad * n.Inputs[i-d].Val
			}
		}
	case Reciprocal:
		n.Inputs[0].Grad += n.Grad * -1 * n.Val * n.Val
	case "":
		if n.IsOutputSymbol() {
			n.Inputs[0].Grad += n.Grad
		}
	}
}

func (n *Node) ComputeVal() {
	switch n.Op {
	case Add:
		n.Val = Sum(Map(n.Inputs, func(n *Node) float64 {
			return n.Val
		}))
	case Multiply:
		n.Val = Product(Map(n.Inputs, func(n *Node) float64 {
			return n.Val
		}))
	case Relu:
		n.Val = Max(0, n.Inputs[0].Val)
	case Exp:
		n.Val = math.Exp(n.Inputs[0].Val)
	case Dot:
		vals := Map(n.Inputs, func(n *Node) float64 {
			return n.Val
		})
		d := len(vals) / 2
		n.Val = DotProduct(vals[:d], vals[d:])
	case Reciprocal:
		n.Val = 1 / n.Inputs[0].Val
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

func ReciprocalNode(label string, outputs [](*Node), input *Node) Node {
	return newNode(label, Reciprocal, [](*Node){input}, outputs)
}

func DotNode(label string, outputs, inputs [](*Node)) Node {
	return newNode(label, Dot, inputs, outputs)
}

func AddNode(label string, outputs, inputs [](*Node)) Node {
	return newNode(label, Add, inputs, outputs)
}

func MultiplyNode(label string, outputs, inputs [](*Node)) Node {
	return newNode(label, Multiply, inputs, outputs)
}

func ReluNode(label string, output, input *Node) Node {
	return newNode(label, Relu, [](*Node){input}, [](*Node){output})
}

func ExpNode(label string, outputs [](*Node), input *Node) Node {
	return newNode(label, Exp, [](*Node){input}, outputs)
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
	Outputs       [](*Node)
	Intermediates [](*Node)
}

func NewGraph(inputs, outputs, intermediates [](*Node)) Graph {
	return Graph{
		Inputs:        inputs,
		Outputs:       outputs,
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
	for i := range g.Outputs {
		g.Outputs[i].Grad = 0
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
	Graph  Graph
	Params [](*Node)
}

func (m *Module) Forward(inputValues []float64, optimizer *Optimizer) (err error) {
	inputValues = append(inputValues, optimizer.GetWeights()...)
	err = m.Graph.Forward(inputValues)
	return
}

func (m *Module) Backprop(upstreamGrads []float64, optimizer *Optimizer) {
	m.Graph.Backprop(upstreamGrads)
	grads := make([]float64, len(m.Params))
	for i := range grads {
		grads[i] = m.Params[i].Grad
	}
	optimizer.UpdateWeights(grads)
}

func NewLinear(n1 int, n2 int, label string) Module {
	inputs := make([](*Node), n1)
	weights := make([](*Node), n1*n2)
	biases := make([](*Node), n2)
	dots := make([](Node), n2)
	dotPts := ToPtrs(dots)
	outputs := make([](*Node), n2)

	// initialize inputs
	for i := 0; i < n1; i++ {
		node := InputSymbol(fmt.Sprintf("%s-input-%d", label, i), dotPts)
		inputs[i] = &node
	}
	unitNode := InputSymbol("unit", dotPts)
	unitNode.Val = 1

	// initialize weights
	for i := 0; i < n2; i++ {
		for j := 0; j < n1; j++ {
			num := j + i*n1
			node := InputSymbol(fmt.Sprintf("%s-weight-%d", label, num), [](*Node){&dots[i]})
			weights[num] = &node
		}
	}

	// initialize biases
	for i := 0; i < n2; i++ {
		node := InputSymbol(fmt.Sprintf("%s-bias-%d", label, i), [](*Node){&dots[i]})
		biases[i] = &node
	}

	// initalize outputs
	for i := 0; i < n2; i++ {
		node := OutputSymbol(fmt.Sprintf("%s-output-%d", label, i), &dots[i])
		outputs[i] = &node
	}

	// initialize dots
	for i := 0; i < n2; i++ {
		dotInputs := append(inputs, &unitNode)
		dotInputs = append(dotInputs, weights[(n1*i):(n1+n1*i)]...)
		dotInputs = append(dotInputs, biases[i])

		dots[i] = DotNode(
			fmt.Sprintf("%s-dot-%d", label, i),
			[](*Node){outputs[i]},
			dotInputs,
		)
	}

	inps := inputs
	for i := 0; i < n2; i++ {
		inps = append(inps, weights[(n1*i):(n1+n1*i)]...)
		inps = append(inps, biases[i])
	}

	return Module{
		Graph:  NewGraph(inps, outputs, dotPts),
		Params: inps[n1:],
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
