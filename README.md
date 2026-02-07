# WebSplat

A fun little 3DGS scene renderer implemented in WebGPU!

## TODO
- [ ] Optimizations, ofc.
- [ ] Try sorting the splats in Raster Renderer and see what happens.
- [X] Bitonic Sort
- [X] Separate NDC Z values into another buffer for faster sort pass? I guess.
- [X] Nothing

---

## How to run

1. Clone the repository
```bash
git clone https://github.com/SP5555/websplat.git
cd websplat
```

2. Install dependencies (You can skip this)
```bash
npm install
```

3. Start local server
```bash
python3 -m http.server 8000
```
or any other static server you prefer.

4. Open in your browser
```bash
http://localhost:8000/
```