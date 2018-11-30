package knn

import (
	"math"
	"sort"
)

type Data struct {
	Name string
	X1 float64
	X2 float64
	X3 float64
	X4 float64
	X5 float64
	Y string
}

type Inference struct {
	Name string
	Distance float64
}

type NearestNeighbors interface {
	Exec(obj Data, train []Data) string
	CalculateDistance(a, b Data) float64
}

type KNN struct {
	K int
}

func NewNearestNeighbors(K int) NearestNeighbors{
	return &KNN{
		K: K,
	}
}

func (k *KNN) CalculateDistance(a, b Data) float64 {
	return math.Sqrt(math.Pow(a.X1 - b.X1, 2)+math.Pow(a.X2 - b.X2, 2)+math.Pow(a.X3 - b.X3, 2)+math.Pow(a.X4 - b.X4, 2)+math.Pow(a.X5 - b.X5, 2))
}

func (k *KNN) Exec(a Data, train []Data) string {
	var dists []Inference
	inf := map[string]int{
		"0":0,
		"1":0,
		"2":0,
		"3":0,
	}

	for t:= 0; t < 600; t++ {
		var dist Inference
		b := train[t]
		dist.Name = b.Y
		dist.Distance = k.CalculateDistance(a, b)
		dists = append(dists, dist)
	}

	sort.Slice(dists, func(x, y int) bool {
		if dists[x].Distance == dists[y].Distance {
			return dists[x].Name < dists[y].Name
		}
		return dists[x].Distance < dists[y].Distance
	})

	for j := 0; j < k.K; j++{
		if dists[j].Name == "0" {
			inf["0"]++;
		} else if dists[j].Name == "1" {
			inf["1"]++;
		} else if dists[j].Name == "2" {
			inf["2"]++;
		} else if dists[j].Name == "3" {
			inf["3"]++;
		}
	}
	m := inf["0"]
	key := "0"
	for index, e := range inf {
		if m < e {
			m = e
			key = index
		}
	}

	return key
}

