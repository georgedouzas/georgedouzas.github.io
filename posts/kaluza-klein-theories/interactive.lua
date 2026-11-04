-- Shortcode {{< interactive <name> [height] >}}: in HTML output embeds the interactive figure
-- figures/interactive/<name>.html (written by figures/<name>.sage: Sage's threejs viewer for 3D scenes,
-- Plotly.js with Sage-written data for 2D plots) in an iframe; in other formats it emits nothing, the static image
-- of the same figure being shown by the usual image block. A missing file aborts the render.
return {
  ["interactive"] = function(args, kwargs, meta)
    local name = pandoc.utils.stringify(args[1])
    local height = args[2] and pandoc.utils.stringify(args[2]) or "480"
    local path = pandoc.path.join({pandoc.path.directory(quarto.doc.input_file), "figures", "interactive", name .. ".html"})
    local f = io.open(path, "r")
    if not f then error("interactive shortcode: no file figures/interactive/" .. name .. ".html") end
    f:close()
    if quarto.doc.is_format("html") then
      return pandoc.RawBlock("html",
        '<div class="interactive-figure"><iframe src="figures/interactive/' .. name .. '.html" width="100%" height="' ..
        height .. '" style="border:1px solid #ddd;border-radius:4px" loading="lazy" title="interactive figure ' ..
        name .. '"></iframe><p class="interactive-note">Interactive figure (drag to rotate or zoom); the static version is above.</p></div>')
    else
      return pandoc.Null()
    end
  end
}
