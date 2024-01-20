package nngo

import (
	"math"
	"math/rand"
	"testing"

	"github.com/stretchr/testify/assert"
)

// f(x,y,z) = (x+y)*z
func TestBackProp1(t *testing.T) {

	var a, b Node

	x := InputSymbol("x", &a)
	y := InputSymbol("y", &a)
	z := InputSymbol("z", &b)
	f := OutputSymbol("f", &b)
	a = AddNode("a", &b, [](*Node){&x, &y})
	b = MultiplyNode("b", &f, &a, &z)

	graph := NewGraph([](*Node){&z, &x, &y}, &f, [](*Node){&a, &b})

	err := graph.Forward([]float64{-4, -2, 5})
	Panic(err)
	assert.True(t, x.Val == -2)
	assert.True(t, y.Val == 5)
	assert.True(t, z.Val == -4)
	assert.True(t, a.Val == 3)
	assert.True(t, b.Val == -12)

	graph.Backprop(1)
	assert.True(t, x.Grad == -4)
	assert.True(t, y.Grad == -4)
	assert.True(t, z.Grad == 3)
	assert.True(t, a.Grad == -4)
	assert.True(t, b.Grad == 1)
}

// f(p,q,r,x,y,z,b) = px + qy + rz + b
func TestBackProp2(t *testing.T) {

	linear := NewLinear(3, "l")
	graph := linear.Graph

	err := graph.Forward([]float64{4, -6, 7, -1, 5, 2, 1})
	Panic(err)

	assert.True(t, graph.Inputs[0].Val == 4)
	assert.True(t, graph.Inputs[1].Val == -6)
	assert.True(t, graph.Inputs[2].Val == 7)
	assert.True(t, graph.Inputs[3].Val == -1)
	assert.True(t, graph.Inputs[4].Val == 5)
	assert.True(t, graph.Inputs[5].Val == 2)
	assert.True(t, graph.Inputs[6].Val == 1)
	assert.True(t, graph.Intermediates[0].Val == -4)
	assert.True(t, graph.Intermediates[1].Val == -30)
	assert.True(t, graph.Intermediates[2].Val == 14)
	assert.True(t, graph.Intermediates[3].Val == -19)
	assert.True(t, graph.Output.Val == -19)

	graph.Backprop(1)
	assert.True(t, graph.Inputs[0].Grad == -1)
	assert.True(t, graph.Inputs[1].Grad == 5)
	assert.True(t, graph.Inputs[2].Grad == 2)
	assert.True(t, graph.Inputs[3].Grad == 4)
	assert.True(t, graph.Inputs[4].Grad == -6)
	assert.True(t, graph.Inputs[5].Grad == 7)
	assert.True(t, graph.Inputs[6].Grad == 1)
	assert.True(t, graph.Intermediates[0].Grad == 1)
	assert.True(t, graph.Intermediates[1].Grad == 1)
	assert.True(t, graph.Intermediates[2].Grad == 1)
	assert.True(t, graph.Intermediates[3].Grad == 1)
	assert.True(t, graph.Output.Grad == 1)
}

// given points (0, 2) and (3, 0), find the equation of line
func TestBackProp3(t *testing.T) {
	linear := NewLinear(2, "l")
	optimizer := NewOptimizer(len(linear.Params), 1e-2, rand.New(rand.NewSource(42)))
	losses := []float64{}
	for i := 0; i < 100; i++ {
		err := linear.Forward([]float64{0, 2}, &optimizer)
		Panic(err)
		loss1 := math.Pow(linear.Graph.Output.Val, 2)
		linear.Backprop(2*linear.Graph.Output.Val, &optimizer)

		err = linear.Forward([]float64{3, 0}, &optimizer)
		Panic(err)
		loss2 := math.Pow(linear.Graph.Output.Val, 2)
		linear.Backprop(2*linear.Graph.Output.Val, &optimizer)

		losses = append(losses, loss1+loss2)
	}
	p := optimizer.params
	assert.True(t, IsStrictlyDecreasing(losses))
	assert.True(t, SimpleFloatEqual(p[0]/p[2], -1./3., 1e-3))
	assert.True(t, SimpleFloatEqual(p[1]/p[2], -1./2., 1e-3))
}
