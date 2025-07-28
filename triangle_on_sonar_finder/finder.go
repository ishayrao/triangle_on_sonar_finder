package triangle_on_sonar_finder

import (
	"context"

	"image"

	"github.com/pkg/errors"
	"go.viam.com/rdk/components/camera"
	"go.viam.com/rdk/logging"
	"go.viam.com/rdk/resource"
	"go.viam.com/rdk/services/vision"
	vis "go.viam.com/rdk/vision"
	"go.viam.com/rdk/vision/classification"
	objdet "go.viam.com/rdk/vision/objectdetection"
	"go.viam.com/rdk/vision/viscapture"
)

const (
	ModelName = "triangle-finder"
)

var (
	Model            = resource.NewModel("viam", "vision", ModelName)
	errUnimplemented = errors.New("unimplemented")
)

func init() {
	resource.RegisterService(vision.API, Model, resource.Registration[vision.Service, *TriangleFinderConfig]{
		Constructor: newTriangleFinder,
	})
}

// TriangleFinderConfig contains the configuration for the triangle finder.
type TriangleFinderConfig struct {
	// Camera is the name of the camera to use for triangle detection.
	Camera string `json:"camera_name"`

	// Threshold is the matching threshold value used for template matching.
	Threshold float32 `json:"threshold,omitempty"`

	// Scale is the resizing scale factor for input images (while maintaining aspect ratio)
	Scale float64 `json:"scale,omitempty"`
}

// TODO: implement Validate
func (cfg TriangleFinderConfig) Validate(path string) ([]string, error) {
	return []string{cfg.Camera}, nil
}

type myTriangleFinder struct {
	resource.AlwaysRebuild

	name      resource.Name
	logger    logging.Logger
	cam       camera.Camera
	config    *TriangleFinderConfig
	templates []TemplateFromImage
	scale     float64
}

func newTriangleFinder(ctx context.Context, deps resource.Dependencies, conf resource.Config, logger logging.Logger) (vision.Service, error) {
	newConf, err := resource.NativeConfig[*TriangleFinderConfig](conf)
	if err != nil {
		return nil, errors.Errorf("failed to parse config for %s got: %s", ModelName, err)
	}

	tf := &myTriangleFinder{
		name:   conf.ResourceName(),
		logger: logger,
		config: newConf,
		scale:  getScaleOrDefault(newConf.Scale),
	}
	// get camera
	tf.cam, err = camera.FromDependencies(deps, newConf.Camera)
	if err != nil {
		return nil, errors.Errorf("failed to get camera from dependencies for %s got: %s", ModelName, err)
	}

	tf.templates, err = loadTemplates(tf.scale)
	if err != nil {
		return nil, errors.Errorf("failed to load template images for %s got: %s", ModelName, err)
	}

	if len(tf.templates) == 0 {
		return nil, errors.Errorf("no valid templates found?!")
	}
	return tf, nil
}

func getScaleOrDefault(scale float64) float64 {
	if scale <= 0 {
		return 0.3 // default value
	}
	return scale
}
func (tf *myTriangleFinder) Name() resource.Name {
	return tf.name
}

func (tf *myTriangleFinder) GetProperties(ctx context.Context, extra map[string]interface{}) (*vision.Properties, error) {
	return &vision.Properties{
		DetectionSupported:      true,
		ClassificationSupported: false,
		ObjectPCDsSupported:     false,
	}, nil
}

func (tf *myTriangleFinder) findTriangles(imgMatrix [][]float64) []objdet.Detection {
	return findTriangles(tf.templates, imgMatrix, 2, tf.config.Threshold, tf.scale)
}

func (tf *myTriangleFinder) DetectionsFromCamera(
	ctx context.Context,
	cameraName string,
	extra map[string]interface{},
) ([]objdet.Detection, error) {
	mimeType := "image/jpeg"
	image, err := camera.DecodeImageFromCamera(ctx, mimeType, nil, tf.cam)
	if err != nil {
		return nil, errors.Errorf("failed to get and decode image for %s got: %s", ModelName, err)
	}

	imgMatrix := ImageToMatrix(image, tf.scale)
	return tf.findTriangles(imgMatrix), nil
}

func (tf *myTriangleFinder) Detections(ctx context.Context, img image.Image, extra map[string]interface{}) ([]objdet.Detection, error) {
	// Convert image to grayscale
	mat := ImageToMatrix(img, tf.scale)
	return tf.findTriangles(mat), nil
}

func (tf *myTriangleFinder) Classifications(ctx context.Context, img image.Image,
	n int, extra map[string]interface{},
) (classification.Classifications, error) {
	return nil, errUnimplemented
}

func (tf *myTriangleFinder) ClassificationsFromCamera(
	ctx context.Context,
	cameraName string,
	n int,
	extra map[string]interface{},
) (classification.Classifications, error) {
	return nil, errUnimplemented
}

func (tf *myTriangleFinder) GetObjectPointClouds(
	ctx context.Context,
	cameraName string,
	extra map[string]interface{},
) ([]*vis.Object, error) {
	return nil, errUnimplemented
}

func (tf *myTriangleFinder) CaptureAllFromCamera(
	ctx context.Context,
	cameraName string,
	opt viscapture.CaptureOptions,
	extra map[string]interface{},
) (viscapture.VisCapture, error) {
	res := viscapture.VisCapture{}
	mimeType := "image/jpeg"
	image, err := camera.DecodeImageFromCamera(ctx, mimeType, nil, tf.cam)
	if err != nil {
		return viscapture.VisCapture{}, errors.Errorf("failed to get image from camera for %s got: %s", ModelName, err)
	}
	if opt.ReturnImage {
		res.Image = image
	}
	if opt.ReturnDetections {
		dets, err := tf.Detections(ctx, image, extra)
		if err != nil {
			return viscapture.VisCapture{}, errors.Errorf("failed to get detections from camera for %s got: %s", ModelName, err)
		}
		res.Detections = dets
	}
	return res, nil
}

func (tf *myTriangleFinder) DoCommand(ctx context.Context, cmd map[string]interface{}) (map[string]interface{}, error) {
	return nil, errUnimplemented
}

func (tf *myTriangleFinder) Close(ctx context.Context) error {
	return nil
}
