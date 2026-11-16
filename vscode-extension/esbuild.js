const esbuild = require('esbuild')

const watch = process.argv.includes('--watch')

const options = {
  entryPoints: ['src/extension.ts'],
  bundle: true,
  outfile: 'dist/extension.js',
  external: ['vscode'],
  format: 'cjs',
  platform: 'node',
  target: 'node18',
  sourcemap: true,
  minify: !watch,
}

async function run() {
  if (watch) {
    const ctx = await esbuild.context(options)
    await ctx.watch()
    console.log('watching...')
  } else {
    await esbuild.build(options)
  }
}

run().catch((err) => {
  console.error(err)
  process.exit(1)
})
