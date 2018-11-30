package main

import (
	"os"
	"encoding/csv"
	"bufio"
	"io"
	"log"
	"github.com/pamungkaski/golang-k-nearest-neighbors"
	"strconv"
	"sort"
	"fmt"
	"time"
	"math/rand"
)

func TakeRandom(data []knn.Data) ([]knn.Data, []knn.Data) {
	r := rand.New(rand.NewSource(time.Now().Unix()))
	val := make([]knn.Data, 200)
	perm := r.Perm(200)
	for i, randIndex := range perm {
		val[i] = data[randIndex]
	}
	for _, i := range perm {
		data[i] = data[len(data)-1] // Replace it with the last one.
		data = data[:len(data)-1]
	}
	return val, data
}

func main()  {
	///// DATA READING
	csvFile, _ := os.Open("DataTrain_Tugas3_AI.csv")
	reader := csv.NewReader(bufio.NewReader(csvFile))
	defer csvFile.Close()
	var data []knn.Data
	reader.Read()
	for  {
		var dt knn.Data
		line, err := reader.Read()
		if err == io.EOF {
			break
		} else if err != nil {
			log.Fatal(err)
		}
		dt.Name = line[0]
		dt.X1, err = strconv.ParseFloat(line[1], 64)
		if err != nil {
			log.Fatal(err)
		}
		dt.X2, err = strconv.ParseFloat(line[2], 64)
		if err != nil {
			log.Fatal(err)
		}
		dt.X3, err = strconv.ParseFloat(line[3], 64)
		if err != nil {
			log.Fatal(err)
		}
		dt.X4, err = strconv.ParseFloat(line[4], 64)
		if err != nil {
			log.Fatal(err)
		}
		dt.X5, err = strconv.ParseFloat(line[5], 64)
		if err != nil {
			log.Fatal(err)
		}
		dt.Y = line[6]
		if err != nil {
			log.Fatal(err)
		}

		data = append(data, dt)
	}
	val, train := TakeRandom(data)
	best := 1
	acc := 0.0
	for i := 1; i <= 200;  i++{
		kalg := knn.NewNearestNeighbors(i)
		right := 0
		for v:= 0; v < 200; v++ {
			var dists []knn.Inference
			a := val[v]
			inf := map[string]int{
				"0":0,
				"1":0,
				"2":0,
				"3":0,
			}

			for t:= 0; t < 600; t++ {
				var dist knn.Inference
				b := train[t]
				dist.Name = b.Y
				dist.Distance = kalg.CalculateDistance(a, b)
				dists = append(dists, dist)
			}

			sort.Slice(dists, func(x, y int) bool {
				if dists[x].Distance == dists[y].Distance {
					return dists[x].Name < dists[y].Name
				}
				return dists[x].Distance < dists[y].Distance
			})

			for j := 0; j < i; j++{
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
			kwey := "0"
			for index, e := range inf {
				if m < e {
					m = e
					kwey = index
				}
			}

			if kwey == a.Y {
				right++
			}
		}
		fmt.Println(i, float64(right)/200.0000)
		if float64(right)/200.0000 > acc {
			acc = float64(right)/200.0000
			best = i
		}
	}

	fmt.Println(best, acc)

}
