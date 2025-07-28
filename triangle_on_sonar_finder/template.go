package triangle_on_sonar_finder

import (
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/png"
	"math"
	"os"

	"golang.org/x/image/font"
	"golang.org/x/image/font/basicfont"
	"golang.org/x/image/math/fixed"

	"github.com/nfnt/resize"
)

// TemplateFromImage represents a template created from an image
type TemplateFromImage struct {
	kernel       [][]float64
	kernelWidth  int
	kernelHeight int
	sumKernel    float32
	originalSize image.Point
}

// NewTemplateFromImage creates a new template from an image file (including preprocessing steps)
func NewTemplateFromImage(img image.Image, scale float64) (*TemplateFromImage, error) {
	originalSize := image.Point{X: img.Bounds().Dx(), Y: img.Bounds().Dy()}
	newWidth := uint(float64(originalSize.X) * scale) // finding new width using same scale as img for resizing
	// step 1: resize template proportionally to how we resize input image
	img = resizeImage(img, newWidth)
	bounds := img.Bounds()
	width := bounds.Dx()
	if width != int(newWidth) {
		return nil, fmt.Errorf("width after resizing (%d) does not match expected newWidth (%d)", width, newWidth)
	}
	height := bounds.Dy()

	kernel := make([][]float64, height)

	for i := range kernel {
		kernel[i] = make([]float64, width) // now kernel is [][]float64
	}

	//step 2: convert image to grayscale matrix
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			// Get grayscale value
			c := img.At(x+bounds.Min.X, y+bounds.Min.Y)
			//using float64 as edge detection requires float for computing the sqrt of sum of squares sqrt(sx*sx + sy*sy)
			grayValue := float64(color.GrayModel.Convert(c).(color.Gray).Y)
			kernel[y][x] = grayValue
		}
	}

	//step 3: applying sobel edge detection
	edgeMatrix := sobelEdge(kernel, width, height, 50)
	edgeKernel := edgeMatrix

	// we do the mean so we're looking for shapes, not color similarity
	// step 4: subtracting mean for shape matching
	var kernelSum float32 = 0
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			kernelSum += float32(edgeKernel[y][x])
		}
	}

	kernelMean := kernelSum / float32(height*width)

	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			edgeKernel[y][x] = float64(float32(edgeKernel[y][x]) - kernelMean)
		}
	}

	var sumKernel float32 = 0
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			sumKernel += float32(edgeKernel[y][x]) * float32(edgeKernel[y][x])
		}
	}

	return &TemplateFromImage{
		kernel:       edgeKernel,
		kernelWidth:  width,
		kernelHeight: height,
		sumKernel:    sumKernel,
		originalSize: originalSize,
	}, nil
}

// FindMatch finds matches of the template in the given image matrix and scales the matches to the original image size
func (t *TemplateFromImage) FindMatch(image [][]float64, stride int, threshold float32, scale float64) []Match {
	height := len(image)
	if height == 0 {
		return nil
	}
	width := len(image[0])

	// Find matches
	var matches []Match
	for i := 0; i < height-t.kernelHeight; i += stride {
		for j := 0; j < width-t.kernelWidth; j += stride {
			// Calculate crop mean
			var cropSum float64 = 0
			for y := 0; y < t.kernelHeight; y++ {
				for x := 0; x < t.kernelWidth; x++ {
					cropSum += image[i+y][j+x]
				}
			}
			cropMean := cropSum / float64(t.kernelHeight*t.kernelWidth)

			sumProduct := 0.0
			sumCropSquared := 0.0

			for y := 0; y < t.kernelHeight; y++ {
				for x := 0; x < t.kernelWidth; x++ {
					normalizedCrop := image[i+y][j+x] - cropMean // mean subtraction from image
					sumProduct += normalizedCrop * t.kernel[y][x]
					sumCropSquared += normalizedCrop * normalizedCrop
				}
			}

			// Calculate correlation coefficient
			denominator := float32(math.Sqrt(float64(float32(sumCropSquared) * t.sumKernel)))
			if denominator > 0 {
				corr := float32(sumProduct) / denominator
				if corr > threshold {
					matches = append(matches, Match{
						X:      int(float64(j) * 1 / scale),
						Y:      int(float64(i) * 1 / scale),
						Width:  t.originalSize.X,
						Height: t.originalSize.Y,
						Score:  corr,
					})
				}
			}
		}
	}

	return matches
}

// Match represents a found match with its position and correlation score
type Match struct {
	X      int
	Y      int
	Width  int
	Height int
	Score  float32
}

// GetBoundingBox returns the bounding box of the match
func (m *Match) GetBoundingBox() image.Rectangle {
	return image.Rectangle{
		Min: image.Point{X: m.X, Y: m.Y},
		Max: image.Point{X: m.X + m.Width, Y: m.Y + m.Height},
	}
}
func resizeImage(img image.Image, newWidth uint) image.Image {
	return resize.Resize(newWidth, 0, img, resize.Lanczos3) //lanczos3 is best for downsampling
}

// uses sobel edge detection for preprocessing of images with different contrast/background colours
func sobelEdge(gray_img [][]float64, width int, height int, threshold int16) [][]float64 {
	edge := make([][]float64, height)
	for y := range edge {
		edge[y] = make([]float64, width)
	}
	// Sobel kernels
	gx := [3][3]int{
		{-1, 0, 1},
		{-2, 0, 2},
		{-1, 0, 1},
	}
	gy := [3][3]int{
		{-1, -2, -1},
		{0, 0, 0},
		{1, 2, 1},
	}
	for y := 1; y < height-1; y++ {
		for x := 1; x < width-1; x++ {
			var sx, sy int
			for ky := -1; ky <= 1; ky++ {
				for kx := -1; kx <= 1; kx++ {
					val := gray_img[y+ky][x+kx]
					sx += int(gx[ky+1][kx+1]) * int(val) //applying sobel kernel to img
					sy += int(gy[ky+1][kx+1]) * int(val)
				}
			}
			edge[y][x] = math.Sqrt(float64(sx*sx + sy*sy)) //computing magnitude of gradient for each pixel using sqrt sum of squares
			if int16(edge[y][x]) < threshold {             //thresholding to remove nose for low contrast edges
				edge[y][x] = 0
			}
		}
	}
	return edge
}

// used for visualizing the edge matrix
func EdgeMatrixToGrayImage(edge [][]float64) *image.Gray {
	height := len(edge)
	width := len(edge[0])
	img := image.NewGray(image.Rect(0, 0, width, height))
	maxVal := 0.0
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			if edge[y][x] > maxVal {
				maxVal = edge[y][x] //finding max val for image normalization
			}
		}
		if maxVal == 0 {
			maxVal = 1
		}
		for y := 0; y < height; y++ {
			for x := 0; x < width; x++ {
				norm := uint8((edge[y][x] / maxVal) * 255)
				img.SetGray(x, y, color.Gray{Y: norm})
			}
		}
	}
	return img
}

// for debugging (show preprocessing steps)
func SaveImageAsPNG(img image.Image, filename string) error {
	f, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer f.Close()
	return png.Encode(f, img)
}
func DrawBoundingBox(img draw.Image, rect image.Rectangle, col color.Color, thickness int, score float32) {
	minX, minY := rect.Min.X, rect.Min.Y
	maxX, maxY := rect.Max.X, rect.Max.Y

	for t := 0; t < thickness; t++ {
		for x := minX + t; x < maxX-t; x++ {
			img.Set(x, minY+t, col)
			img.Set(x, maxY-1-t, col)
		}
		for y := minY + t; y < maxY-t; y++ {
			img.Set(minX+t, y, col)
			img.Set(maxX-1-t, y, col)
		}

	}
	// label boxes with detection score
	label := fmt.Sprintf("%.2f", score)
	point := fixed.Point26_6{
		X: fixed.I(minX),
		Y: fixed.I(minY - 2), // above box
	}
	d := &font.Drawer{
		Dst:  img,
		Src:  image.NewUniform(color.RGBA{255, 0, 0, 255}),
		Face: basicfont.Face7x13,
		Dot:  point,
	}
	d.DrawString(label)
}
