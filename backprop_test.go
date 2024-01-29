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

	graph := NewGraph([](*Node){&z, &x, &y}, [](*Node){&f}, [](*Node){&a, &b})

	err := graph.Forward([]float64{-4, -2, 5})
	Panic(err)
	assert.Equal(t, -2.0, x.Val)
	assert.Equal(t, 5.0, y.Val)
	assert.Equal(t, -4.0, z.Val)
	assert.Equal(t, 3.0, a.Val)
	assert.Equal(t, -12.0, b.Val)

	graph.Backprop([]float64{1})
	assert.Equal(t, -4.0, x.Grad)
	assert.Equal(t, -4.0, y.Grad)
	assert.Equal(t, 3.0, z.Grad)
	assert.Equal(t, -4.0, a.Grad)
	assert.Equal(t, 1.0, b.Grad)

}

// f(p,q,r,x,y,z,b) = px + qy + rz + b
func TestBackProp2(t *testing.T) {

	linear := NewLinear(3, 1, "l")
	graph := linear.Graph

	err := graph.Forward([]float64{4, -6, 7, -1, 5, 2, 1})
	Panic(err)

	assert.Equal(t, 4.0, graph.Inputs[0].Val)
	assert.Equal(t, -6.0, graph.Inputs[1].Val)
	assert.Equal(t, 7.0, graph.Inputs[2].Val)
	assert.Equal(t, -1.0, graph.Inputs[3].Val)
	assert.Equal(t, 5.0, graph.Inputs[4].Val)
	assert.Equal(t, 2.0, graph.Inputs[5].Val)
	assert.Equal(t, 1.0, graph.Inputs[6].Val)
	assert.Equal(t, -19.0, graph.Intermediates[0].Val)
	assert.Equal(t, -19.0, graph.Outputs[0].Val)

	graph.Backprop([]float64{1})
	assert.Equal(t, -1.0, graph.Inputs[0].Grad)
	assert.Equal(t, 5.0, graph.Inputs[1].Grad)
	assert.Equal(t, 2.0, graph.Inputs[2].Grad)
	assert.Equal(t, 4.0, graph.Inputs[3].Grad)
	assert.Equal(t, -6.0, graph.Inputs[4].Grad)
	assert.Equal(t, 7.0, graph.Inputs[5].Grad)
	assert.Equal(t, 1.0, graph.Inputs[6].Grad)
	assert.Equal(t, 1.0, graph.Intermediates[0].Grad)
	assert.Equal(t, 1.0, graph.Outputs[0].Grad)

}

// given points (0, 2) and (3, 0), find the equation of line
func TestBackProp3(t *testing.T) {
	linear := NewLinear(2, 1, "l")
	optimizer := NewOptimizer(len(linear.Params), 1e-2, rand.New(rand.NewSource(42)))
	losses := []float64{}
	for i := 0; i < 100; i++ {
		err := linear.Forward([]float64{0, 2}, &optimizer)
		Panic(err)
		loss1 := math.Pow(linear.Graph.Outputs[0].Val, 2)
		linear.Graph.ZeroGrad()
		linear.Backprop([]float64{2 * linear.Graph.Outputs[0].Val}, &optimizer)

		err = linear.Forward([]float64{3, 0}, &optimizer)
		Panic(err)
		loss2 := math.Pow(linear.Graph.Outputs[0].Val, 2)
		linear.Graph.ZeroGrad()
		linear.Backprop([]float64{2 * linear.Graph.Outputs[0].Val}, &optimizer)

		losses = append(losses, loss1+loss2)
	}
	p := optimizer.params
	assert.True(t, IsNonIncreasing(losses))
	assert.InEpsilon(t, -1./3., p[0]/p[2], 1e-3)
	assert.InEpsilon(t, -1./2., p[1]/p[2], 1e-3)
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

	graph := NewGraph([](*Node){&x, &y}, [](*Node){&f}, [](*Node){&a, &b, &c, &d, &e})

	err := graph.Forward([]float64{2, 3})
	Panic(err)
	assert.Equal(t, 2.0, x.Val)
	assert.Equal(t, 3.0, y.Val)
	assert.Equal(t, 5.0, a.Val)
	assert.Equal(t, 6.0, b.Val)
	assert.Equal(t, 11.0, c.Val)
	assert.Equal(t, 30.0, d.Val)
	assert.Equal(t, 41.0, e.Val)
	assert.Equal(t, 41.0, f.Val)

	graph.Backprop([]float64{1})
	assert.Equal(t, 1.0, f.Grad)
	assert.Equal(t, 1.0, e.Grad)
	assert.Equal(t, 1.0, d.Grad)
	assert.Equal(t, 1.0, c.Grad)
	assert.Equal(t, 6.0, b.Grad)
	assert.Equal(t, 7.0, a.Grad)
	assert.Equal(t, 25.0, x.Grad)
	assert.Equal(t, 19.0, y.Grad)

}

// f(x) = x * x
func TestBackProp5(t *testing.T) {
	var a Node
	x := InputSymbol("x", [](*Node){&a})
	f := OutputSymbol("f", &a)
	a = MultiplyNode("a", [](*Node){&f}, [](*Node){&x, &x})

	graph := NewGraph([](*Node){&x}, [](*Node){&f}, [](*Node){&a})

	err := graph.Forward([]float64{3})
	Panic(err)
	assert.Equal(t, 3.0, x.Val)
	assert.Equal(t, 9.0, a.Val)
	assert.Equal(t, 9.0, f.Val)

	graph.Backprop([]float64{1})
	assert.Equal(t, 1.0, f.Grad)
	assert.Equal(t, 1.0, a.Grad)
	assert.Equal(t, 6.0, x.Grad)
}

// f(x) = x * x * x
func TestBackProp6(t *testing.T) {
	var a Node
	x := InputSymbol("x", [](*Node){&a})
	f := OutputSymbol("f", &a)
	a = MultiplyNode("a", [](*Node){&f}, [](*Node){&x, &x, &x})

	graph := NewGraph([](*Node){&x}, [](*Node){&f}, [](*Node){&a})

	err := graph.Forward([]float64{2})
	Panic(err)
	assert.Equal(t, 2.0, x.Val)
	assert.Equal(t, 8.0, a.Val)
	assert.Equal(t, 8.0, f.Val)

	graph.Backprop([]float64{1})
	assert.Equal(t, 1.0, f.Grad)
	assert.Equal(t, 1.0, a.Grad)
	assert.Equal(t, 12.0, x.Grad)
}

// f(x) = relu(x + y)
func TestBackProp7(t *testing.T) {
	var a, b Node
	x := InputSymbol("x", [](*Node){&a})
	y := InputSymbol("y", [](*Node){&a})
	f := OutputSymbol("f", &b)
	a = AddNode("a", [](*Node){&b}, [](*Node){&x, &y})
	b = ReluNode("b", &f, &a)

	graph := NewGraph([](*Node){&x, &y}, [](*Node){&f}, [](*Node){&a, &b})

	err := graph.Forward([]float64{2, 3})
	Panic(err)
	assert.Equal(t, 2.0, x.Val)
	assert.Equal(t, 3.0, y.Val)
	assert.Equal(t, 5.0, a.Val)
	assert.Equal(t, 5.0, b.Val)
	assert.Equal(t, 5.0, f.Val)

	graph.Backprop([]float64{1})
	assert.Equal(t, 1.0, f.Grad)
	assert.Equal(t, 1.0, b.Grad)
	assert.Equal(t, 1.0, a.Grad)
	assert.Equal(t, 1.0, y.Grad)
	assert.Equal(t, 1.0, x.Grad)

	graph.ZeroGrad()

	err = graph.Forward([]float64{2, -5})
	Panic(err)
	assert.Equal(t, 2.0, x.Val)
	assert.Equal(t, -5.0, y.Val)
	assert.Equal(t, -3.0, a.Val)
	assert.Equal(t, 0.0, b.Val)
	assert.Equal(t, 0.0, f.Val)

	graph.Backprop([]float64{1})
	assert.Equal(t, 1.0, f.Grad)
	assert.Equal(t, 1.0, b.Grad)
	assert.Equal(t, 0.0, a.Grad)
	assert.Equal(t, 0.0, y.Grad)
	assert.Equal(t, 0.0, x.Grad)

}

// f(x) = exp(x)
func TestBackprop8(t *testing.T) {
	var a Node
	x := InputSymbol("x", [](*Node){&a})
	f1 := OutputSymbol("f1", &a)
	f2 := OutputSymbol("f2", &a)
	a = ExpNode("a", [](*Node){&f1, &f2}, &x)

	graph := NewGraph([](*Node){&x}, [](*Node){&f1, &f2}, [](*Node){&a})

	err := graph.Forward([]float64{1})
	Panic(err)

	assert.Equal(t, 1.0, x.Val)
	assert.InDelta(t, math.Exp(1), a.Val, 1e-3)
	assert.InDelta(t, math.Exp(1), f1.Val, 1e-3)
	assert.InDelta(t, math.Exp(1), f2.Val, 1e-3)

	graph.Backprop([]float64{1, 1})
	assert.Equal(t, 1.0, f1.Grad)
	assert.Equal(t, 1.0, f2.Grad)
	assert.Equal(t, 2.0, a.Grad)
	assert.InDelta(t, 2*math.Exp(1), x.Grad, 1e-3)
}

/*
2x + y = 5 passes through (0, 5) and (2.5, 0)
x - 3y = 3 passes through (0, -1) and (3, 0)
*/
func TestBackProp9(t *testing.T) {
	linear := NewLinear(2, 2, "l")
	optimizer := NewOptimizer(len(linear.Params), 1e-2, rand.New(rand.NewSource(42)))
	losses1 := []float64{}
	losses2 := []float64{}
	for i := 0; i < 100; i++ {
		localLoss := []float64{}
		for _, point := range [][]float64{{0, 5}, {2.5, 0}} {
			err := linear.Forward(point, &optimizer)
			Panic(err)
			localLoss = append(localLoss, math.Pow(linear.Graph.Outputs[0].Val, 2))
			linear.Graph.ZeroGrad()
			linear.Backprop([]float64{2 * linear.Graph.Outputs[0].Val, 0.}, &optimizer)
		}
		losses1 = append(losses1, Sum(localLoss))

		localLoss = []float64{}
		for _, point := range [][]float64{{0, -1.}, {3, 0}} {
			err := linear.Forward(point, &optimizer)
			Panic(err)
			localLoss = append(localLoss, math.Pow(linear.Graph.Outputs[1].Val, 2))
			linear.Graph.ZeroGrad()
			linear.Backprop([]float64{0., 2 * linear.Graph.Outputs[1].Val}, &optimizer)
		}
		losses2 = append(losses2, Sum(localLoss))
	}
	p := optimizer.params
	assert.True(t, IsNonIncreasing(losses1))
	assert.True(t, IsNonIncreasing(losses2))

	assert.InDelta(t, -2./5., p[0]/p[2], 5e-2)
	assert.InDelta(t, -1./5., p[1]/p[2], 5e-2)
	assert.InDelta(t, -1./3., p[3]/p[5], 5e-2)
	assert.InDelta(t, 1., p[4]/p[5], 5e-2)
}

// f([x1, ..., xn], [y1, ..., yn]) = Dot(x, y)
func TestBackProp10(t *testing.T) {
	var a Node
	x1 := InputSymbol("x1", [](*Node){&a})
	x2 := InputSymbol("x2", [](*Node){&a})
	x3 := InputSymbol("x3", [](*Node){&a})

	y1 := InputSymbol("y1", [](*Node){&a})
	y2 := InputSymbol("y2", [](*Node){&a})
	y3 := InputSymbol("y3", [](*Node){&a})

	f := OutputSymbol("f", &a)

	a = DotNode("a", [](*Node){&f}, [](*Node){&x1, &x2, &x3, &y1, &y2, &y3})

	graph := NewGraph([](*Node){&x1, &x2, &x3, &y1, &y2, &y3}, [](*Node){&f}, [](*Node){&a})

	err := graph.Forward([]float64{1, 2, 3, -2, 4, -1})
	Panic(err)

	assert.Equal(t, 1.0, x1.Val)
	assert.Equal(t, 2.0, x2.Val)
	assert.Equal(t, 3.0, x3.Val)
	assert.Equal(t, -2.0, y1.Val)
	assert.Equal(t, 4.0, y2.Val)
	assert.Equal(t, -1.0, y3.Val)
	assert.Equal(t, 3.0, a.Val)
	assert.Equal(t, 3.0, f.Val)

	graph.Backprop([]float64{1})
	assert.Equal(t, 1.0, f.Grad)
	assert.Equal(t, 1.0, a.Grad)
	assert.Equal(t, -2.0, x1.Grad)
	assert.Equal(t, 4.0, x2.Grad)
	assert.Equal(t, -1.0, x3.Grad)
	assert.Equal(t, 1.0, y1.Grad)
	assert.Equal(t, 2.0, y2.Grad)
	assert.Equal(t, 3.0, y3.Grad)
}

// f(x) = 1/x
func TestBackProp11(t *testing.T) {
	var a Node
	x := InputSymbol("x", [](*Node){&a})
	f1 := OutputSymbol("f1", &a)
	f2 := OutputSymbol("f2", &a)
	a = ReciprocalNode("a", [](*Node){&f1, &f2}, &x)

	graph := NewGraph([](*Node){&x}, [](*Node){&f1, &f2}, [](*Node){&a})

	err := graph.Forward([]float64{5})
	Panic(err)

	assert.Equal(t, 5.0, x.Val)
	assert.Equal(t, 0.2, a.Val)
	assert.Equal(t, 0.2, f1.Val)
	assert.Equal(t, 0.2, f2.Val)

	graph.Backprop([]float64{1, 1})
	assert.Equal(t, 1.0, f1.Grad)
	assert.Equal(t, 1.0, f2.Grad)
	assert.Equal(t, 2.0, a.Grad)
	assert.InDelta(t, -0.08, x.Grad, 1e-3)
}

func TestBackprop12(t *testing.T) {
	s := SoftMax(3, "s")
	err := s.Forward([]float64{1, 2, 3})
	Panic(err)

	expSum := math.Exp(1) + math.Exp(2) + math.Exp(3)

	assert.Equal(t, 1., s.Inputs[0].Val)
	assert.Equal(t, 2., s.Inputs[1].Val)
	assert.Equal(t, 3., s.Inputs[2].Val)
	assert.InEpsilon(t, math.Exp(1), s.Intermediates[0].Val, 1e-6)
	assert.InEpsilon(t, math.Exp(2), s.Intermediates[1].Val, 1e-6)
	assert.InEpsilon(t, math.Exp(3), s.Intermediates[2].Val, 1e-6)
	assert.InEpsilon(t, expSum, s.Intermediates[3].Val, 1e-6)
	assert.InEpsilon(t, 1./expSum, s.Intermediates[4].Val, 1e-6)
	assert.InEpsilon(t, math.Exp(1)/expSum, s.Intermediates[5].Val, 1e-6)
	assert.InEpsilon(t, math.Exp(2)/expSum, s.Intermediates[6].Val, 1e-6)
	assert.InEpsilon(t, math.Exp(3)/expSum, s.Intermediates[7].Val, 1e-6)
	assert.InEpsilon(t, math.Exp(1)/expSum, s.Outputs[0].Val, 1e-6)
	assert.InEpsilon(t, math.Exp(2)/expSum, s.Outputs[1].Val, 1e-6)
	assert.InEpsilon(t, math.Exp(3)/expSum, s.Outputs[2].Val, 1e-6)
}

func TestBackprop13(t *testing.T) {
	s1 := SoftMax(3, "s1")
	s2 := SoftMax(3, "s2")
	s := Merge([]Graph{s1, s2})
	err := s.Forward([]float64{1, 2, 3})
	Panic(err)
	expSum := math.Exp(1) + math.Exp(2) + math.Exp(3)
	expSum2 := math.Exp(math.Exp(1)/expSum) + math.Exp(math.Exp(2)/expSum) + math.Exp(math.Exp(3)/expSum)
	assert.Equal(t, 1., s.Inputs[0].Val)
	assert.Equal(t, 2., s.Inputs[1].Val)
	assert.Equal(t, 3., s.Inputs[2].Val)
	assert.InEpsilon(t, math.Exp(1), s.Intermediates[0].Val, 1e-6)
	assert.InEpsilon(t, math.Exp(2), s.Intermediates[1].Val, 1e-6)
	assert.InEpsilon(t, math.Exp(3), s.Intermediates[2].Val, 1e-6)
	assert.InEpsilon(t, expSum, s.Intermediates[3].Val, 1e-6)
	assert.InEpsilon(t, 1./expSum, s.Intermediates[4].Val, 1e-6)
	assert.InEpsilon(t, math.Exp(1)/expSum, s.Intermediates[5].Val, 1e-6)
	assert.InEpsilon(t, math.Exp(2)/expSum, s.Intermediates[6].Val, 1e-6)
	assert.InEpsilon(t, math.Exp(3)/expSum, s.Intermediates[7].Val, 1e-6)
	assert.InEpsilon(t, math.Exp(1)/expSum, s.Intermediates[8].Val, 1e-6)
	assert.InEpsilon(t, math.Exp(2)/expSum, s.Intermediates[9].Val, 1e-6)
	assert.InEpsilon(t, math.Exp(3)/expSum, s.Intermediates[10].Val, 1e-6)
	assert.InEpsilon(t, math.Exp(1)/expSum, s.Intermediates[11].Val, 1e-6)
	assert.InEpsilon(t, math.Exp(2)/expSum, s.Intermediates[12].Val, 1e-6)
	assert.InEpsilon(t, math.Exp(3)/expSum, s.Intermediates[13].Val, 1e-6)
	assert.InEpsilon(t, math.Exp(math.Exp(1)/expSum), s.Intermediates[14].Val, 1e-6)
	assert.InEpsilon(t, math.Exp(math.Exp(2)/expSum), s.Intermediates[15].Val, 1e-6)
	assert.InEpsilon(t, math.Exp(math.Exp(3)/expSum), s.Intermediates[16].Val, 1e-6)
	assert.InEpsilon(t, expSum2, s.Intermediates[17].Val, 1e-6)
	assert.InEpsilon(t, 1./expSum2, s.Intermediates[18].Val, 1e-6)
	assert.InEpsilon(t, math.Exp(math.Exp(1)/expSum)/expSum2, s.Intermediates[19].Val, 1e-6)
	assert.InEpsilon(t, math.Exp(math.Exp(2)/expSum)/expSum2, s.Intermediates[20].Val, 1e-6)
	assert.InEpsilon(t, math.Exp(math.Exp(3)/expSum)/expSum2, s.Intermediates[21].Val, 1e-6)
	assert.InEpsilon(t, math.Exp(math.Exp(1)/expSum)/expSum2, s.Outputs[0].Val, 1e-6)
	assert.InEpsilon(t, math.Exp(math.Exp(2)/expSum)/expSum2, s.Outputs[1].Val, 1e-6)
	assert.InEpsilon(t, math.Exp(math.Exp(3)/expSum)/expSum2, s.Outputs[2].Val, 1e-6)
}
