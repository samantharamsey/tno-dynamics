(function() {
    function parse(buffer) {
        const view = new DataView(buffer);
        if (view.getUint32(0, false) !== 0x5a565342 || view.getUint32(4, true) !== 1) {
            throw new Error('invalid preview mesh bank');
        }
        const count = view.getUint32(8, true);
        const frames = [];
        let offset = 12;
        for (let frameIndex = 0; frameIndex < count; frameIndex++) {
            const slider = view.getFloat32(offset, true);
            const C = view.getFloat64(offset + 4, true);
            const counts = [
                [view.getUint32(offset + 12, true), view.getUint32(offset + 16, true)],
                [view.getUint32(offset + 20, true), view.getUint32(offset + 24, true)]
            ];
            offset += 28;
            const meshes = [];
            for (const [vertices, faces] of counts) {
                const mesh = {};
                for (const key of ['x', 'y', 'z']) {
                    mesh[key] = new Float32Array(buffer, offset, vertices);
                    offset += vertices*4;
                }
                for (const key of ['i', 'j', 'k']) {
                    mesh[key] = new Uint16Array(buffer, offset, faces);
                    offset += faces*2;
                }
                meshes.push(mesh);
                offset = (offset + 3) & ~3;
            }
            offset = (offset + 3) & ~3;
            frames.push({slider: slider, C: C, meshes: meshes});
        }
        return frames;
    }

    async function load() {
        if (typeof DecompressionStream === 'undefined') return null;
        const response = await fetch('/assets/preview_frames.dat');
        if (!response.ok) return null;
        const compressed = await response.arrayBuffer();
        const stream = new Blob([compressed]).stream().pipeThrough(new DecompressionStream('gzip'));
        return parse(await new Response(stream).arrayBuffer());
    }

    window.neptuneZvsPreviewPromise = load().catch(function(error) {
        console.error('could not load local preview meshes', error);
        return null;
    });
})();
