package nngo

import (
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

	graph := NewGraph([](*Node){&x, &y, &z}, &f, [](*Node){&a, &b})

	err := graph.Forward([]float64{-2, 5, -4})
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

// f(a,b,c,x,y,z) = ax + by + cz
func TestBackProp2(t *testing.T) {

	l := NewLinear(3, "l")
	graph := l.Graph

	err := graph.Forward([]float64{4, -6, 7, -1, 5, 2})
	Panic(err)

	assert.True(t, graph.Inputs[0].Val == 4)
	assert.True(t, graph.Inputs[1].Val == -6)
	assert.True(t, graph.Inputs[2].Val == 7)
	assert.True(t, graph.Inputs[3].Val == -1)
	assert.True(t, graph.Inputs[4].Val == 5)
	assert.True(t, graph.Inputs[5].Val == 2)
	assert.True(t, graph.Intermediates[0].Val == -4)
	assert.True(t, graph.Intermediates[1].Val == -30)
	assert.True(t, graph.Intermediates[2].Val == 14)
	assert.True(t, graph.Intermediates[3].Val == -20)
	assert.True(t, graph.Output.Val == -20)

	graph.Backprop(1)
	assert.True(t, graph.Inputs[0].Grad == -1)
	assert.True(t, graph.Inputs[1].Grad == 5)
	assert.True(t, graph.Inputs[2].Grad == 2)
	assert.True(t, graph.Inputs[3].Grad == 4)
	assert.True(t, graph.Inputs[4].Grad == -6)
	assert.True(t, graph.Inputs[5].Grad == 7)
	assert.True(t, graph.Intermediates[0].Grad == 1)
	assert.True(t, graph.Intermediates[1].Grad == 1)
	assert.True(t, graph.Intermediates[2].Grad == 1)
	assert.True(t, graph.Intermediates[3].Grad == 1)
	assert.True(t, graph.Output.Grad == 1)
}
