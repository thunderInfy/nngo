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

	x := InputSymbol("x", [](*Node){&a})
	y := InputSymbol("y", [](*Node){&a})
	z := InputSymbol("z", [](*Node){&b})
	f := OutputSymbol("f", &b)
	a = AddNode("a", [](*Node){&b}, [](*Node){&x, &y})
	b = MultiplyNode("b", [](*Node){&f}, [](*Node){&a, &z})

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
		linear.Graph.ZeroGrad()
		linear.Backprop(2*linear.Graph.Output.Val, &optimizer)

		err = linear.Forward([]float64{3, 0}, &optimizer)
		Panic(err)
		loss2 := math.Pow(linear.Graph.Output.Val, 2)
		linear.Graph.ZeroGrad()
		linear.Backprop(2*linear.Graph.Output.Val, &optimizer)

		losses = append(losses, loss1+loss2)
	}
	p := optimizer.params
	assert.True(t, IsStrictlyDecreasing(losses))
	assert.True(t, SimpleFloatEqual(p[0]/p[2], -1./3., 1e-3))
	assert.True(t, SimpleFloatEqual(p[1]/p[2], -1./2., 1e-3))
}

/*
T([x, y]) = [x+y, x*y]
U = T(T([x, y]))
f = U[0] + U[1]
*/
func TestBackProp4(t *testing.T) {

	var a, b, c, d, e Node

	x := InputSymbol("x", [](*Node){&a, &b})
	y := InputSymbol("y", [](*Node){&a, &b})
	f := OutputSymbol("f", &e)
	a = AddNode("a", [](*Node){&c, &d}, [](*Node){&x, &y})
	b = MultiplyNode("b", [](*Node){&c, &d}, [](*Node){&x, &y})
	c = AddNode("c", [](*Node){&e}, [](*Node){&a, &b})
	d = MultiplyNode("d", [](*Node){&e}, [](*Node){&a, &b})
	e = AddNode("e", [](*Node){&f}, [](*Node){&c, &d})

	graph := NewGraph([](*Node){&x, &y}, &f, [](*Node){&a, &b, &c, &d, &e})

	err := graph.Forward([]float64{2, 3})
	Panic(err)
	assert.True(t, x.Val == 2)
	assert.True(t, y.Val == 3)
	assert.True(t, a.Val == 5)
	assert.True(t, b.Val == 6)
	assert.True(t, c.Val == 11)
	assert.True(t, d.Val == 30)
	assert.True(t, e.Val == 41)
	assert.True(t, f.Val == 41)

	graph.Backprop(1)
	assert.True(t, graph.Output.Grad == 1)
	assert.True(t, graph.Intermediates[4].Grad == 1)
	assert.True(t, graph.Intermediates[3].Grad == 1)
	assert.True(t, graph.Intermediates[2].Grad == 1)
	assert.True(t, graph.Intermediates[1].Grad == 6)
	assert.True(t, graph.Intermediates[0].Grad == 7)
	assert.True(t, graph.Inputs[0].Grad == 25)
	assert.True(t, graph.Inputs[1].Grad == 19)
}

// f(x) = x * x
func TestBackProp5(t *testing.T) {
	var a Node
	x := InputSymbol("x", [](*Node){&a})
	f := OutputSymbol("f", &a)
	a = MultiplyNode("a", [](*Node){&f}, [](*Node){&x, &x})

	graph := NewGraph([](*Node){&x}, &f, [](*Node){&a})

	err := graph.Forward([]float64{3})
	Panic(err)
	assert.True(t, x.Val == 3)
	assert.True(t, a.Val == 9)
	assert.True(t, f.Val == 9)

	graph.Backprop(1)
	assert.True(t, graph.Output.Grad == 1)
	assert.True(t, graph.Intermediates[0].Grad == 1)
	assert.True(t, graph.Inputs[0].Grad == 6)
}

// f(x) = x * x * x
func TestBackProp6(t *testing.T) {
	var a Node
	x := InputSymbol("x", [](*Node){&a})
	f := OutputSymbol("f", &a)
	a = MultiplyNode("a", [](*Node){&f}, [](*Node){&x, &x, &x})

	graph := NewGraph([](*Node){&x}, &f, [](*Node){&a})

	err := graph.Forward([]float64{2})
	Panic(err)
	assert.True(t, x.Val == 2)
	assert.True(t, a.Val == 8)
	assert.True(t, f.Val == 8)

	graph.Backprop(1)
	assert.True(t, graph.Output.Grad == 1)
	assert.True(t, graph.Intermediates[0].Grad == 1)
	assert.True(t, graph.Inputs[0].Grad == 12)
}
