import express from 'express';
import multer from 'multer';
import path from 'path';
import { fileURLToPath } from 'url';
import fs from 'fs';
import { PythonShell } from 'python-shell';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const port = 3000;

// Configure multer for image upload
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    const uploadDir = path.join(__dirname, 'uploads');
    const outputDir = path.join(__dirname, 'outputs');
    
    // Create directories if they don't exist
    [uploadDir, outputDir].forEach(dir => {
      if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
      }
    });
    
    cb(null, uploadDir);
  },
  filename: (req, file, cb) => {
    const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
    const ext = path.extname(file.originalname);
    cb(null, 'image-' + uniqueSuffix + ext);
  }
});

const upload = multer({ 
  storage,
  limits: { fileSize: 10 * 1024 * 1024 }, // 10MB max
  fileFilter: (req, file, cb) => {
    const allowedTypes = /jpeg|jpg|png|gif|webp/;
    const ext = allowedTypes.test(path.extname(file.originalname).toLowerCase());
    const mimetype = allowedTypes.test(file.mimetype);
    if (ext && mimetype) return cb(null, true);
    cb(new Error('Only image files are allowed!'));
  }
});

// Enable CORS
app.use((req, res, next) => {
  res.header('Access-Control-Allow-Origin', '*');
  res.header('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept');
  next();
});

// Serve static files
app.use('/uploads', express.static(path.join(__dirname, 'uploads')));
app.use('/outputs', express.static(path.join(__dirname, 'outputs')));

// Main detection route
app.post('/api/detect', upload.single('image'), (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: 'No image uploaded' });
  }

  const inputImagePath = req.file.path;
  const outputImagePath = path.join(__dirname, 'outputs', path.basename(inputImagePath));

  // Call Python detection script with explicit python3 path
  PythonShell.run('server/detect.py', {
    mode: 'text',
    pythonPath: 'python3',
    args: [inputImagePath, outputImagePath]
  }, function (err, results) {
    if (err) {
      console.error('Python error:', err);
      return res.status(500).json({ error: 'Failed to process image' });
    }

    try {
      const detections = JSON.parse(results[0]);
      
      // Check if there was an error from Python script
      if (detections.error) {
        return res.status(500).json({ error: detections.error });
      }

      return res.json({
        originalImage: `http://localhost:${port}/uploads/${path.basename(inputImagePath)}`,
        processedImage: `http://localhost:${port}/outputs/${path.basename(inputImagePath)}`,
        detectedObjects: detections
      });
    } catch (parseError) {
      console.error('JSON parse error:', parseError);
      return res.status(500).json({ error: 'Invalid detection result from Python' });
    }
  });
});

app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}`);
});