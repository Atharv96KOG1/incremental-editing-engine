import java.io.PrintWriter
importCpg("__CPG_PATH__")

def jsonEsc(s: String): String = s.replace("\\", "\\\\").replace("\"", "\\\"")

val edges = cpg.call
  .filterNot(c => c.name.startsWith("<operator>") || (c.name.startsWith("__") && c.name.endsWith("__")))
  .map(c => (c.method.name, c.method.filename, c.name, c.lineNumber.getOrElse(-1)))
  .l
  .distinct

val pw = new PrintWriter("__OUT_PATH__")
pw.println("[")
pw.println(edges.map { case (callerName, callerFile, calleeName, line) =>
  s"""  {"caller": "${jsonEsc(callerName)}", "file": "${jsonEsc(callerFile)}", "callee": "${jsonEsc(calleeName)}", "line": $line}"""
}.mkString(",\n"))
pw.println("]")
pw.close()
