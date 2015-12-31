import scala.util.Random

object Test {
  def main(args: Array[String]) {
    val a = Random.nextLong()

    println(s"a = $a")

    val rnd1 = new Random()
    //rnd1.setSeed(a)

    println(s"rnd1.next = ${rnd1.nextLong()}")

    val rnd2 = new Random()
    //rnd2.setSeed(a)

    println(s"rnd2.next = ${rnd2.nextLong()}")
  }
}