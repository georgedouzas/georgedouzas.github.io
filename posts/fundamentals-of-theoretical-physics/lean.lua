-- Shortcode {{< lean <name> >}}: shows a Lean statement from this post's lean-statements.json.
local statements = nil
local function here() return pandoc.path.directory(quarto.doc.input_file) end
local function load()
  if statements then return statements end
  local f = io.open(pandoc.path.join({here(), "lean-statements.json"}), "r")
  if not f then error("lean shortcode: lean-statements.json not found") end
  local text = f:read("a"); f:close()
  statements = quarto.json.decode(text)
  return statements
end
return {
  ["lean"] = function(args, kwargs, meta)
    local name = pandoc.utils.stringify(args[1])
    local s = load()[name]
    if not s then error("lean shortcode: no statement named " .. name) end
    local body = (s.doc ~= "" and (s.doc .. "\n") or "") .. s.statement
    local where = "`" .. name .. "` in `" .. s.file .. "`"
    if s.namespace ~= "" then where = where .. ", namespace `" .. s.namespace .. "`" end
    return pandoc.Div({
      pandoc.CodeBlock(body, pandoc.Attr("", {"lean"})),
      pandoc.Para(pandoc.read(where, "markdown").blocks[1].content),
    }, pandoc.Attr("", {"lean-statement"}))
  end
}
