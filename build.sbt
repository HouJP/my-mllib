name := "GBoost"

version := "1.0"

scalaVersion := "2.10.4"

libraryDependencies += "org.apache.spark" %% "spark-core" % "1.5.0"
libraryDependencies += "org.apache.spark" %% "spark-mllib" % "1.5.0"
libraryDependencies += "com.github.scopt" %% "scopt" % "3.3.0"
libraryDependencies += "org.scalatest" %% "scalatest" % "2.2.1" % "test"

resolvers += Resolver.sonatypeRepo("public")

assemblyMergeStrategy in assembly := {
  case PathList("akka", xs @ _*) => MergeStrategy.last
  case PathList("com", "google", "common", "base", xs @_*) => MergeStrategy.first
  case PathList("org", "apache", xs @ _*) => MergeStrategy.first
  case PathList("javax", "xml", xs @ _*) => MergeStrategy.first
  case PathList(ps @ _*) if ps.last endsWith "Log$Logger.class" => MergeStrategy.first
  case PathList(ps @ _*) if ps.last endsWith "Log.class" => MergeStrategy.first
  case x =>
    val oldStrategy = (assemblyMergeStrategy in assembly).value
    oldStrategy(x)
}